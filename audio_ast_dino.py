#!/usr/bin/env python3
"""
DINO-style self-supervised learning for AST-shaped audio transformers.

Stage 1:
- self-supervised DINO pretraining on `data/audioset_5000` plus unlabeled ESC-50

Stage 2:
- supervised finetuning on ESC-50 using the shared AudioLoader split
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import ConcatDataset
from transformers import AutoConfig, AutoModel
from sklearn.metrics import classification_report, confusion_matrix

from common.dataloader import AudioLoader, ESC50_AUDIO_RATE
from common.lib import (
    SEED,
    seed_everything,
    make_training_args,
    compute_metrics,
    percent_trained,
    balance_classes,
)
from novel.dino_utils import (
    DINOHead,
    compute_feature_std,
    dino_pretrain_from_datasets,
    dino_finetune_from_datasets,
)
from novel.mae_lib import evaluate_knn_and_tsne_on_test
from audio_ast_mae import (
    AST_MODEL_NAME,
    find_audio_files,
    LogMelPadCrop,
    FolderAudioPretrainDataset,
    WaveformArrayPretrainDataset,
    WaveformArrayClassificationDataset,
    _extract_last_hidden_state,
)


class AudioASTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.hidden_size = int(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        hidden = _extract_last_hidden_state(outputs)
        return hidden.mean(dim=1)


class AudioASTDINO(nn.Module):
    def __init__(self, config, out_dim: int = 256):
        super().__init__()
        self.base_config = config

        self.student_backbone = AudioASTBackbone(config)
        self.student_head = DINOHead(self.student_backbone.hidden_size, out_dim=out_dim)

        self.teacher_backbone = AudioASTBackbone(copy.deepcopy(config))
        self.teacher_head = DINOHead(self.teacher_backbone.hidden_size, out_dim=out_dim)

        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        for ps, pt in zip(
            self.student_backbone.parameters(), self.teacher_backbone.parameters()
        ):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor, momentum: float = 0.9) -> None:
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(momentum).add_(batch_center, alpha=1 - momentum)

    def dino_loss(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
    ) -> torch.Tensor:
        student_logp = F.log_softmax(student_out / student_temp, dim=-1)
        teacher_prob = F.softmax((teacher_out.detach() - self.center) / teacher_temp, dim=-1)
        return torch.sum(-teacher_prob * student_logp, dim=-1).mean()

    def forward(self, x1=None, x2=None, **kwargs):
        if x1 is None:
            x1 = kwargs.get("x1")
        if x2 is None:
            x2 = kwargs.get("x2")
        if x1 is None or x2 is None:
            raise ValueError("AudioASTDINO expects x1 and x2")

        student_feat = self.student_backbone(x1)
        student_out = self.student_head(student_feat)

        with torch.no_grad():
            teacher_feat = self.teacher_backbone(x2)
            teacher_out = self.teacher_head(teacher_feat)

        loss = self.dino_loss(student_out, teacher_out)

        with torch.no_grad():
            self.update_center(teacher_out)

        return {"loss": loss, "logits": student_out}

    def build_classifier(
        self,
        n_classes: int,
        class_weights: torch.Tensor | None = None,
    ):
        backbone_config = self.base_config.__class__.from_dict(self.base_config.to_dict())
        backbone = AutoModel.from_config(backbone_config)
        backbone.load_state_dict(self.student_backbone.model.state_dict(), strict=False)
        return AudioASTDINOClassifier(
            backbone=backbone,
            hidden_size=self.student_backbone.hidden_size,
            n_classes=n_classes,
            class_weights=class_weights,
        )


class AudioASTDINOClassifier(nn.Module):
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
        pooled = hidden.mean(dim=1)
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


class AudioDINOPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        if torch.rand(1).item() < 0.8:
            x = x + 0.05 * torch.randn_like(x)
        if torch.rand(1).item() < 0.5:
            shift = int(torch.randint(-32, 33, (1,)).item())
            x = torch.roll(x, shifts=shift, dims=0)
        if torch.rand(1).item() < 0.5:
            scale = 0.8 + 0.4 * torch.rand(1).item()
            x = x * scale
        if torch.rand(1).item() < 0.3:
            width = min(x.shape[0] // 8, 64)
            if width > 0:
                start = int(torch.randint(0, max(1, x.shape[0] - width + 1), (1,)).item())
                x[start : start + width] = 0.0
        return x

    def __getitem__(self, idx: int):
        x = self.base_dataset[idx]["x"]
        return {"x1": self.augment(x), "x2": self.augment(x)}


def dino_collator(batch):
    x1 = torch.stack([b["x1"] for b in batch])
    x2 = torch.stack([b["x2"] for b in batch])
    return {"x1": x1, "x2": x2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audioset_dir", type=str, default="data/audioset_5000")
    parser.add_argument("--model_name", type=str, default=AST_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="data/audio_ast_dino_runs")
    parser.add_argument("--teacher_momentum", type=float, default=0.996)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=10)
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

    full_dataset = ConcatDataset(datasets)
    valid_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - valid_size
    base_train_dataset, base_valid_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    dino_train_dataset = AudioDINOPretrainDataset(base_train_dataset)
    dino_valid_dataset = AudioDINOPretrainDataset(base_valid_dataset)

    config = AutoConfig.from_pretrained(args.model_name)
    dino_model = AudioASTDINO(config=config, out_dim=args.out_dim)

    if args.checkpoint is None:
        pretrain_args = make_training_args(
            output_dir=str(Path(args.output_dir) / "dino_pretrain"),
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
            lr=args.pretrain_lr,
            seed=SEED,
        )
        pretrain_args.load_best_model_at_end = False
        pretrain_args.metric_for_best_model = None
        pretrain_args.weight_decay = args.weight_decay

        dino_trainer = dino_pretrain_from_datasets(
            train_dataset=dino_train_dataset,
            valid_dataset=dino_valid_dataset,
            dino_model=dino_model,
            training_args=pretrain_args,
            teacher_momentum=args.teacher_momentum,
            feature_std_dataset=base_train_dataset,
        )
        print("DINO validation:")
        print(dino_trainer.evaluate())
    else:
        state_dict = load_file(args.checkpoint)
        dino_model.load_state_dict(state_dict)

    mean_std = compute_feature_std(
        dino_model.student_backbone,
        base_train_dataset,
        n_samples=min(2000, len(base_train_dataset)),
    )
    print(f"Post-pretrain — Mean feature std: {mean_std:.4f}")

    def _dino_get_embeddings(X_np: np.ndarray) -> np.ndarray:
        dino_model.eval()
        with torch.no_grad():
            device = next(dino_model.parameters()).device
            embeddings = []
            for waveform in X_np:
                spec = transform(torch.tensor(waveform, dtype=torch.float32), ESC50_AUDIO_RATE)
                x = spec.to(device, dtype=torch.float32).unsqueeze(0)
                z = dino_model.student_backbone(x)
                embeddings.append(z.cpu())
            return torch.cat(embeddings, dim=0).numpy()

    idx2cls = {i: label for i, label in enumerate(esc50_data["label_names"])}
    evaluate_knn_and_tsne_on_test(
        X_train=esc50_data["X_train"],
        y_train=esc50_data["y_train"],
        X_test=esc50_data["X_test"],
        y_test=esc50_data["y_test"],
        get_embeddings=_dino_get_embeddings,
        idx2cls=idx2cls,
        output_dir=args.output_dir,
        prefix="audio_dino",
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
    if args.balance_target_size is not None:
        X_train, y_train = balance_classes(
            X_train,
            y_train,
            target_size=args.balance_target_size,
            seed=SEED,
            n_classes=n_classes,
        )

    train_dataset = WaveformArrayClassificationDataset(
        X_train, y_train, source_rate=ESC50_AUDIO_RATE, transform=transform
    )
    valid_dataset = WaveformArrayClassificationDataset(
        X_valid, y_valid, source_rate=ESC50_AUDIO_RATE, transform=transform
    )
    test_dataset = WaveformArrayClassificationDataset(
        X_test, y_test, source_rate=ESC50_AUDIO_RATE, transform=transform
    )

    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights_np = class_counts.sum() / (len(class_counts) * class_counts + 1e-8)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    clf_model = dino_model.build_classifier(
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

    clf_trainer, val_metrics, test_metrics = dino_finetune_from_datasets(
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
