#!/usr/bin/env python3
"""
MAE-style pretraining for AST-shaped audio spectrogram transformers.

This script combines two unlabeled sources:
- raw audio files under `data/audioset_5000`
- ESC-50 clips loaded through `common.dataloader.AudioLoader`

The model is initialized from a Hugging Face AST/SSAST config (non-pretrained),
then trained with a masked patch reconstruction objective in the same spirit as
`mae.py` / `mae_freq.py`.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import ConcatDataset
from transformers import AutoConfig, AutoModel, Trainer

from safetensors.torch import load_file

from common.dataloader import AudioLoader, ESC50_AUDIO_RATE
from common.lib import (
    SEED,
    seed_everything,
    make_training_args,
    compute_metrics,
    percent_trained,
    balance_classes,
)
from mae_freq import SpectrogramLayout, mae_forward_loss
from novel.mae_lib import (
    mae_collator,
    cls_collator,
    evaluate_knn_and_tsne_on_test,
)
from sklearn.metrics import classification_report, confusion_matrix


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
AST_MODEL_NAME = "Simon-Kotchou/ssast-small-patch-audioset-16-16"


def find_audio_files(root: str | None) -> list[str]:
    if root is None:
        return []

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    files = [
        str(path)
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    ]
    return sorted(set(files))


def compatible_num_heads(embed_dim: int, preferred_heads: int) -> int:
    preferred_heads = max(1, int(preferred_heads))
    if embed_dim % preferred_heads == 0:
        return preferred_heads

    gcd_heads = math.gcd(embed_dim, preferred_heads)
    if gcd_heads > 0:
        return gcd_heads

    for heads in range(min(embed_dim, preferred_heads), 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


def _extract_last_hidden_state(outputs):
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


class LogMelPadCrop:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        win_length: int = 400,
        hop_length: int = 160,
        target_length: int = 1024,
        dataset_mean: float = -4.2677393,
        dataset_std: float = 4.5689974,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        spec = self.melspec(waveform)
        spec = self.amplitude_to_db(spec)
        spec = spec.squeeze(0).transpose(0, 1)

        t, _ = spec.shape
        if t < self.target_length:
            pad = self.target_length - t
            spec = torch.nn.functional.pad(spec, (0, 0, 0, pad))
        elif t > self.target_length:
            start = random.randint(0, t - self.target_length)
            spec = spec[start : start + self.target_length]

        spec = (spec - self.dataset_mean) / (self.dataset_std * 2.0)
        return spec


class FolderAudioPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, files: list[str], transform: LogMelPadCrop):
        self.files = self._filter_decodable_files(files)
        self.transform = transform
        if not self.files:
            raise RuntimeError("No decodable audio files found for pretraining.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        waveform, sr = torchaudio.load(self.files[idx])
        spec = self.transform(waveform, sr)
        return {"x": spec}

    @staticmethod
    def _filter_decodable_files(files: list[str]) -> list[str]:
        valid_files = []
        skipped = 0
        for path in files:
            try:
                waveform, _ = torchaudio.load(path)
                if waveform.numel() == 0:
                    skipped += 1
                    continue
                valid_files.append(path)
            except Exception:
                skipped += 1
        if skipped:
            print(f"Skipped {skipped} unreadable audio files during dataset scan.")
        return valid_files


class WaveformArrayPretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        waveforms: np.ndarray,
        source_rate: int,
        transform: LogMelPadCrop,
    ):
        self.waveforms = waveforms
        self.source_rate = source_rate
        self.transform = transform

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int):
        waveform = torch.tensor(self.waveforms[idx], dtype=torch.float32)
        spec = self.transform(waveform, self.source_rate)
        return {"x": spec}


class WaveformArrayClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        waveforms: np.ndarray,
        labels: np.ndarray,
        source_rate: int,
        transform: LogMelPadCrop,
    ):
        self.waveforms = waveforms
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.source_rate = source_rate
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        waveform = torch.tensor(self.waveforms[idx], dtype=torch.float32)
        spec = self.transform(waveform, self.source_rate)
        return {"x": spec, "labels": self.labels[idx]}


def random_mask_by_count(
    x: torch.Tensor,
    mask_patch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, D = x.shape
    if mask_patch <= 0 or mask_patch >= N:
        raise ValueError(f"mask_patch must be in [1, {N - 1}], got {mask_patch}")

    len_keep = N - mask_patch
    noise = torch.rand(B, N, device=x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]

    x_masked = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
    )

    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class AudioASTMAE(nn.Module):
    def __init__(
        self,
        config,
        num_mel_bins: int,
        target_length: int,
        fshape: int,
        tshape: int,
        mask_patch: int,
        decoder_dim: int = 256,
    ):
        super().__init__()

        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.mask_patch = mask_patch
        self.base_config = config
        self.hidden_size = int(config.hidden_size)
        self.decoder_dim = decoder_dim

        self.layout = SpectrogramLayout(
            freq_bins=(num_mel_bins // fshape) * fshape,
            time_frames=(target_length // tshape) * tshape,
            freq_patch=fshape,
            time_patch=tshape,
        )
        self.num_patches = self.layout.num_patches
        self.patch_size = self.layout.patch_area

        self.backbone = AutoModel.from_config(config)

        self.decoder_embed = nn.Linear(self.hidden_size, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        decoder_heads = compatible_num_heads(
            decoder_dim,
            max(1, int(getattr(config, "num_attention_heads", 8)) // 2),
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_size)
        self.loss_fn = nn.MSELoss(reduction="none")

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        x = x[:, : self.layout.time_frames, : self.layout.freq_bins]
        Tp = self.layout.num_time_patches
        Fp = self.layout.num_freq_patches
        x = x.reshape(B, Tp, self.layout.time_patch, Fp, self.layout.freq_patch)
        x = x.permute(0, 1, 3, 2, 4)
        return x.reshape(B, Tp * Fp, self.patch_size)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, : self.layout.time_frames, : self.layout.freq_bins]
        embeddings = self.backbone.embeddings(x)
        special_tokens = embeddings.shape[1] - self.num_patches
        if special_tokens < 0:
            raise ValueError(
                f"Backbone returned too few tokens: got {embeddings.shape[1]}, expected at least {self.num_patches}"
            )
        return embeddings[:, special_tokens:, :]

    def forward_encoder(self, x: torch.Tensor):
        target = self.patchify(x).detach()
        tokens = self.tokenize(x)
        x_masked, mask, ids_restore = random_mask_by_count(tokens, self.mask_patch)
        encoder_outputs = self.backbone.encoder(x_masked)
        latent = _extract_last_hidden_state(encoder_outputs)
        if hasattr(self.backbone, "layernorm"):
            latent = self.backbone.layernorm(latent)
        return latent, target, mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(latent)
        B, N_keep, D = x.shape
        N = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(B, N - N_keep, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )
        x_ = x_ + self.decoder_pos
        x_ = self.decoder(x_)
        x_ = self.decoder_norm(x_)
        return self.decoder_pred(x_)

    def forward(self, x=None, labels=None):
        latent, target, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        loss = mae_forward_loss(self.loss_fn, target, pred, mask)
        return {"loss": loss, "logits": pred}

    def build_classifier(
        self,
        n_classes: int,
        class_weights: torch.Tensor | None = None,
    ):
        backbone_config = self.base_config.__class__.from_dict(self.base_config.to_dict())
        backbone = AutoModel.from_config(backbone_config)
        backbone.load_state_dict(self.backbone.state_dict(), strict=False)
        model = AudioASTClassifier(
            backbone=backbone,
            hidden_size=self.hidden_size,
            n_classes=n_classes,
            class_weights=class_weights,
        )
        return model


class AudioASTClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        n_classes: int,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, n_classes)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x=None, labels=None):
        outputs = self.backbone(x)
        hidden = _extract_last_hidden_state(outputs)
        if hidden.dim() != 3:
            raise ValueError(f"Expected hidden states with shape [B, N, D], got {hidden.shape}")
        pooled = hidden.mean(dim=1)
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


def mae_pretrain_from_datasets(
    train_dataset,
    valid_dataset,
    mae_model: nn.Module,
    training_args,
):
    trainer = Trainer(
        model=mae_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=mae_collator,
    )
    trainer.train()
    return trainer


def mae_finetune_from_datasets(
    clf_model: nn.Module,
    train_dataset,
    valid_dataset,
    test_dataset,
    training_args,
):
    trainer = Trainer(
        model=clf_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=cls_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    return trainer, val_metrics, test_metrics


def get_audio_ast_embeddings(
    backbone: nn.Module,
    transform: LogMelPadCrop,
    waveforms: np.ndarray,
    source_rate: int,
) -> np.ndarray:
    backbone.eval()
    device = next(backbone.parameters()).device
    embeddings = []
    with torch.no_grad():
        for waveform in waveforms:
            spec = transform(torch.tensor(waveform, dtype=torch.float32), source_rate)
            x = spec.to(device, dtype=torch.float32).unsqueeze(0)
            outputs = backbone(x)
            hidden = _extract_last_hidden_state(outputs)
            embeddings.append(hidden.mean(dim=1).cpu())
    return torch.cat(embeddings, dim=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audioset_dir", type=str, default="data/audioset_5000")
    parser.add_argument("--model_name", type=str, default=AST_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="data/audio_ast_mae_runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pretrain_batch_size", type=int, default=8)
    parser.add_argument("--finetune_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrain_lr", type=float, default=1e-4)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_mel_bins", type=int, default=128)
    parser.add_argument("--target_length", type=int, default=1024)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--dataset_mean", type=float, default=-4.2677393)
    parser.add_argument("--dataset_std", type=float, default=4.5689974)
    parser.add_argument("--fshape", type=int, default=16)
    parser.add_argument("--tshape", type=int, default=16)
    parser.add_argument("--mask_patch", type=int, default=100)
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument(
        "--percent_train",
        type=float,
        default=100.0,
        help="Percent of labeled training data to use for finetuning (0-100)",
    )
    parser.add_argument("--balance_target_size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    seed_everything(SEED)

    transform = LogMelPadCrop(
        sample_rate=args.sample_rate,
        n_mels=args.num_mel_bins,
        target_length=args.target_length,
        dataset_mean=args.dataset_mean,
        dataset_std=args.dataset_std,
    )

    datasets = []

    audioset_files = find_audio_files(args.audioset_dir)
    if audioset_files:
        datasets.append(FolderAudioPretrainDataset(audioset_files, transform))
        print(f"Found {len(audioset_files)} audio files in {args.audioset_dir}")
    else:
        print(f"No audio files found in {args.audioset_dir}, skipping that source.")

    esc50_data = AudioLoader(args).load()
    esc50_waveforms = np.concatenate(
        [esc50_data["X_train"], esc50_data["X_valid"], esc50_data["X_test"]],
        axis=0,
    )
    datasets.append(
        WaveformArrayPretrainDataset(
            esc50_waveforms,
            source_rate=ESC50_AUDIO_RATE,
            transform=transform,
        )
    )
    print(f"Loaded {len(esc50_waveforms)} ESC-50 clips from AudioLoader.")

    if not datasets:
        raise RuntimeError("No audio sources available for pretraining.")

    full_dataset = ConcatDataset(datasets)
    valid_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    config = AutoConfig.from_pretrained(args.model_name)
    mae_model = AudioASTMAE(
        config=config,
        num_mel_bins=args.num_mel_bins,
        target_length=args.target_length,
        fshape=args.fshape,
        tshape=args.tshape,
        mask_patch=args.mask_patch,
        decoder_dim=args.decoder_dim,
    )

    idx2cls = {i: label for i, label in enumerate(esc50_data["label_names"])}

    print("\n=== Zero-shot KNN/t-SNE before MAE pretraining ===")
    evaluate_knn_and_tsne_on_test(
        X_train=esc50_data["X_train"],
        y_train=esc50_data["y_train"],
        X_test=esc50_data["X_test"],
        y_test=esc50_data["y_test"],
        get_embeddings=lambda X_np: get_audio_ast_embeddings(
            mae_model.backbone,
            transform,
            X_np,
            ESC50_AUDIO_RATE,
        ),
        idx2cls=idx2cls,
        output_dir=args.output_dir,
        prefix="audio_mae_before",
    )

    pretrain_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "pretrain"),
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        lr=args.pretrain_lr,
        seed=SEED,
    )
    pretrain_args.metric_for_best_model = "eval_loss"
    pretrain_args.greater_is_better = False
    pretrain_args.weight_decay = args.weight_decay

    if args.checkpoint is None:
        trainer = mae_pretrain_from_datasets(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            mae_model=mae_model,
            training_args=pretrain_args,
        )

        print("Pretraining validation metrics:")
        print(trainer.evaluate())

        pretrain_dir = Path(args.output_dir) / "pretrain_final"
        pretrain_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(pretrain_dir))
        torch.save(mae_model.state_dict(), pretrain_dir / "audio_ast_mae.pt")
    else:
        state_dict = load_file(args.checkpoint)
        mae_model.load_state_dict(state_dict)

    print("\n=== Zero-shot KNN/t-SNE after MAE pretraining ===")
    evaluate_knn_and_tsne_on_test(
        X_train=esc50_data["X_train"],
        y_train=esc50_data["y_train"],
        X_test=esc50_data["X_test"],
        y_test=esc50_data["y_test"],
        get_embeddings=lambda X_np: get_audio_ast_embeddings(
            mae_model.backbone,
            transform,
            X_np,
            ESC50_AUDIO_RATE,
        ),
        idx2cls=idx2cls,
        output_dir=args.output_dir,
        prefix="audio_mae_after",
    )

    print("\n=== Stage 2: audio classifier finetuning ===")
    X_train = esc50_data["X_train"]
    X_valid = esc50_data["X_valid"]
    X_test = esc50_data["X_test"]
    y_train = esc50_data["y_train"]
    y_valid = esc50_data["y_valid"]
    y_test = esc50_data["y_test"]
    label_names = esc50_data["label_names"]
    n_classes = int(esc50_data["n_classes"])

    X_train, y_train = percent_trained(X_train, y_train, args)

    balance_target_size = args.balance_target_size
    if balance_target_size is not None:
        X_train, y_train = balance_classes(
            X_train,
            y_train,
            target_size=balance_target_size,
            seed=SEED,
            n_classes=n_classes,
        )

    train_dataset = WaveformArrayClassificationDataset(
        X_train,
        y_train,
        source_rate=ESC50_AUDIO_RATE,
        transform=transform,
    )
    valid_dataset = WaveformArrayClassificationDataset(
        X_valid,
        y_valid,
        source_rate=ESC50_AUDIO_RATE,
        transform=transform,
    )
    test_dataset = WaveformArrayClassificationDataset(
        X_test,
        y_test,
        source_rate=ESC50_AUDIO_RATE,
        transform=transform,
    )

    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights_np = class_counts.sum() / (len(class_counts) * class_counts + 1e-8)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    clf_model = mae_model.build_classifier(
        n_classes=n_classes,
        class_weights=class_weights,
    )

    finetune_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "finetune"),
        epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        lr=args.finetune_lr,
        seed=SEED,
    )
    finetune_args.metric_for_best_model = "macro_f1"
    finetune_args.greater_is_better = True

    clf_trainer, val_metrics, test_metrics = mae_finetune_from_datasets(
        clf_model=clf_model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        training_args=finetune_args,
    )

    print("Validation metrics:")
    print(val_metrics)
    print("Test metrics:")
    print(test_metrics)

    pred_output = clf_trainer.predict(test_dataset)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids
    label_ids = list(range(n_classes))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=label_names,
            zero_division=0,
        )
    )
    print(confusion_matrix(y_true, y_pred, labels=label_ids))


if __name__ == "__main__":
    main()
