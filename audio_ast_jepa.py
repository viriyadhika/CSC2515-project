#!/usr/bin/env python3
"""
JEPA-style pretraining for AST-shaped audio spectrogram transformers.

This mirrors the operational flow of `audio_ast_mae.py`:
- same ESC-50 / AudioSet data sources
- same epoch-0 and milestone representation evaluation
- same temporary finetuning on ESC-50

The methodological difference is the SSL objective:
- MAE reconstructs masked spectrogram patches
- JEPA predicts latent target-patch representations in embedding space
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.utils.data import ConcatDataset, random_split
from transformers import AutoConfig, AutoModel

from common.dataloader import AudioLoader, ESC50_AUDIO_RATE
from common.lib import (
    SEED,
    balance_classes,
    make_training_args,
    percent_trained,
    seed_everything,
)
from evaluation.embedding_eval import evaluate_embedding_snapshots
from mae_freq import SpectrogramLayout
from models.audio_common import (
    AST_MODEL_NAME,
    FolderAudioPretrainDataset,
    LogMelPadCrop,
    WaveformArrayClassificationDataset,
    WaveformArrayPretrainDataset,
    extract_last_hidden_state,
    find_audio_files,
)
from novel.mae_lib import mae_collator
from training.pretrain_loop import run_finetune, run_pretrain_loop


class AudioASTJEPA(nn.Module):
    def __init__(
        self,
        config,
        num_mel_bins: int,
        target_length: int,
        fshape: int,
        tshape: int,
        mask_ratio: float = 0.6,
        pred_hidden_dim: int = 512,
    ):
        super().__init__()

        self.base_config = config
        self.hidden_size = int(config.hidden_size)
        self.mask_ratio = mask_ratio

        self.layout = SpectrogramLayout(
            freq_bins=(num_mel_bins // fshape) * fshape,
            time_frames=(target_length // tshape) * tshape,
            freq_patch=fshape,
            time_patch=tshape,
        )
        self.num_patches = self.layout.num_patches

        self.backbone = AutoModel.from_config(config)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_size, pred_hidden_dim),
            nn.GELU(),
            nn.Linear(pred_hidden_dim, self.hidden_size),
        )
        self.loss_fn = nn.MSELoss()

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, : self.layout.time_frames, : self.layout.freq_bins]
        embeddings = self.backbone.embeddings(x)
        special_tokens = embeddings.shape[1] - self.num_patches
        if special_tokens < 0:
            raise ValueError(
                f"Backbone returned too few tokens: got {embeddings.shape[1]}, "
                f"expected at least {self.num_patches}"
            )

        encoder_outputs = self.backbone.encoder(embeddings)
        latent = extract_last_hidden_state(encoder_outputs)
        if hasattr(self.backbone, "layernorm"):
            latent = self.backbone.layernorm(latent)
        return latent[:, special_tokens:, :]

    def random_mask(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, _ = x.shape
        num_mask = min(n_tokens, max(1, int(n_tokens * self.mask_ratio)))
        mask = torch.zeros(bsz, n_tokens, dtype=torch.bool, device=x.device)
        for row in range(bsz):
            indices = torch.randperm(n_tokens, device=x.device)[:num_mask]
            mask[row, indices] = True
        return mask

    def forward(self, x=None, labels=None):
        if x is None:
            raise ValueError("AudioASTJEPA.forward expects input tensor x")

        latent = self.encode_patches(x)
        mask = self.random_mask(latent)

        target_latent = latent.detach()
        pred_latent = self.predictor(latent)

        target_masked = target_latent[mask]
        pred_masked = pred_latent[mask]
        loss = self.loss_fn(pred_masked, target_masked)

        return {"loss": loss, "logits": pred_masked}

    def clone_backbone(self) -> nn.Module:
        return copy.deepcopy(self.backbone)


def load_checkpoint_into_model(model, checkpoint_path: str) -> None:
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audioset_dir", type=str, default="data/audioset_5000")
    parser.add_argument("--model_name", type=str, default=AST_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="data/audio_ast_jepa_runs")
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=15)
    parser.add_argument("--pretrain_batch_size", type=int, default=8)
    parser.add_argument("--finetune_batch_size", type=int, default=8)
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
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--pred_hidden_dim", type=int, default=512)
    parser.add_argument("--percent_train", type=float, default=100.0)
    parser.add_argument("--balance_target_size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--esc50_only_pretrain", action="store_true")
    parser.add_argument("--run_final_finetune", action="store_true")
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
    audioset_files = []
    if args.esc50_only_pretrain:
        print("ESC50-only pretraining enabled, skipping AudioSet source.")
    else:
        audioset_files = find_audio_files(args.audioset_dir)
        if audioset_files:
            datasets.append(FolderAudioPretrainDataset(audioset_files, transform))
            print(f"Found {len(audioset_files)} audio files in {args.audioset_dir}")
        else:
            print(f"No audio files found in {args.audioset_dir}, skipping that source.")

    esc50_data = AudioLoader(args).load()
    esc50_waveforms = esc50_data["X_train"]
    datasets.append(
        WaveformArrayPretrainDataset(
            esc50_waveforms,
            source_rate=ESC50_AUDIO_RATE,
            transform=transform,
        )
    )
    print(f"Loaded {len(esc50_waveforms)} ESC-50 train clips from AudioLoader.")

    if not datasets:
        raise RuntimeError("No audio sources available for pretraining.")

    full_dataset = ConcatDataset(datasets)
    valid_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - valid_size
    pretrain_train_dataset, pretrain_valid_dataset = random_split(
        full_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    finetune_data = copy.deepcopy(esc50_data)
    x_train, y_train = percent_trained(
        finetune_data["X_train"], finetune_data["y_train"], args
    )
    if args.balance_target_size is not None:
        x_train, y_train = balance_classes(
            x_train,
            y_train,
            target_size=args.balance_target_size,
            seed=SEED,
            n_classes=int(finetune_data["n_classes"]),
        )
    finetune_data["X_train"] = x_train
    finetune_data["y_train"] = y_train

    embedding_train_dataset = WaveformArrayClassificationDataset(
        finetune_data["X_train"],
        finetune_data["y_train"],
        source_rate=ESC50_AUDIO_RATE,
        transform=transform,
    )
    embedding_test_dataset = WaveformArrayClassificationDataset(
        finetune_data["X_test"],
        finetune_data["y_test"],
        source_rate=ESC50_AUDIO_RATE,
        transform=transform,
    )

    config = AutoConfig.from_pretrained(args.model_name)
    jepa_model = AudioASTJEPA(
        config=config,
        num_mel_bins=args.num_mel_bins,
        target_length=args.target_length,
        fshape=args.fshape,
        tshape=args.tshape,
        mask_ratio=args.mask_ratio,
        pred_hidden_dim=args.pred_hidden_dim,
    )
    if args.checkpoint is not None:
        load_checkpoint_into_model(jepa_model, args.checkpoint)
        print(f"Loaded model from checkpoint: {args.checkpoint}")

    idx2cls = {i: label for i, label in enumerate(finetune_data["label_names"])}

    initial_dir = Path(args.output_dir) / "epoch_0"
    initial_snapshot = evaluate_embedding_snapshots(
        backbone=jepa_model.clone_backbone(),
        train_dataset=embedding_train_dataset,
        y_train=finetune_data["y_train"],
        test_dataset=embedding_test_dataset,
        y_test=finetune_data["y_test"],
        output_dir=initial_dir,
        idx2cls=idx2cls,
        batch_size=max(args.pretrain_batch_size, args.finetune_batch_size),
    )
    print(f"Initial KNN accuracy: {initial_snapshot['knn_accuracy']:.4f}")

    def evaluate_epoch(model, epoch: int) -> None:
        epoch_dir = Path(args.output_dir) / f"epoch_{epoch}"
        backbone = model.clone_backbone()
        snapshot = evaluate_embedding_snapshots(
            backbone=backbone,
            train_dataset=embedding_train_dataset,
            y_train=finetune_data["y_train"],
            test_dataset=embedding_test_dataset,
            y_test=finetune_data["y_test"],
            output_dir=epoch_dir,
            idx2cls=idx2cls,
            batch_size=max(args.pretrain_batch_size, args.finetune_batch_size),
        )
        print(f"Epoch {epoch} KNN accuracy: {snapshot['knn_accuracy']:.4f}")

        finetune_args = make_training_args(
            output_dir=str(epoch_dir),
            epochs=args.finetune_epochs,
            batch_size=args.finetune_batch_size,
            lr=args.finetune_lr,
            seed=SEED,
        )
        finetune_args.metric_for_best_model = "macro_f1"
        finetune_args.greater_is_better = True
        metrics = run_finetune(
            backbone=backbone,
            esc50_data=finetune_data,
            training_args=finetune_args,
            transform=transform,
            output_dir=epoch_dir,
        )
        print(f"Epoch {epoch} finetune validation metrics: {metrics['val_metrics']}")
        print(f"Epoch {epoch} finetune test metrics: {metrics['test_metrics']}")

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

    if args.pretrain_epochs > 0:
        run_pretrain_loop(
            model=jepa_model,
            train_dataset=pretrain_train_dataset,
            valid_dataset=pretrain_valid_dataset,
            collator=mae_collator,
            training_args=pretrain_args,
            eval_callback=evaluate_epoch,
        )

    if args.run_final_finetune and args.finetune_epochs > 0:
        final_dir = Path(args.output_dir) / "final_finetune"
        final_args = make_training_args(
            output_dir=str(final_dir),
            epochs=args.finetune_epochs,
            batch_size=args.finetune_batch_size,
            lr=args.finetune_lr,
            seed=SEED,
        )
        final_args.metric_for_best_model = "macro_f1"
        final_args.greater_is_better = True
        metrics = run_finetune(
            backbone=jepa_model.clone_backbone(),
            esc50_data=finetune_data,
            training_args=final_args,
            transform=transform,
            output_dir=final_dir,
        )
        print(f"Final finetune validation metrics: {metrics['val_metrics']}")
        print(f"Final finetune test metrics: {metrics['test_metrics']}")


if __name__ == "__main__":
    main()
