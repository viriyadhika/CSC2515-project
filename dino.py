#!/usr/bin/env python3
"""
DINO-style self-supervised pretraining for ECG, then finetuning for classification.

Pipeline:
1) Extract 198-sample beats and RR from MIT-BIH (same as MAE / TinyTransformer)
2) DINO pretraining with student/teacher and momentum encoder
3) Finetune classifier (TinyTransformerMAEFinetune) initialized from DINO student backbone
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainerCallback

from common.lib import (
    SEED,
    IDX2CLS,
    seed_everything,
    extract_beats_and_rr,
    normalize_rows,
    maybe_augment_noise,
    ECGRRDataset,
    make_training_args,
    compute_metrics,
    percent_trained,
)
from novel.mae_lib import cls_collator, add_common_ecg_cli_args


class ECGEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int = 198,
        patch_size: int = 9,
        embed_dim: int = 64,
        nhead: int = 8,
        n_layer: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv1d(
            1, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layer)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).transpose(1, 2)  # [B, N, D]
        x = x + self.pos[:, : x.size(1), :]
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global embedding [B, D]
        return x


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden_dim: int = 512,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class ECGDINO(nn.Module):
    def __init__(
        self,
        seq_len: int = 198,
        patch_size: int = 9,
        embed_dim: int = 64,
        nhead: int = 8,
        n_layer: int = 4,
        out_dim: int = 256,
    ):
        super().__init__()

        self.student_backbone = ECGEncoder(
            seq_len=seq_len,
            patch_size=patch_size,
            embed_dim=embed_dim,
            nhead=nhead,
            n_layer=n_layer,
        )
        self.student_head = DINOHead(embed_dim, out_dim=out_dim)

        self.teacher_backbone = ECGEncoder(
            seq_len=seq_len,
            patch_size=patch_size,
            embed_dim=embed_dim,
            nhead=nhead,
            n_layer=n_layer,
        )

        self.teacher_head = DINOHead(embed_dim, out_dim=out_dim)

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
        for ps, pt in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
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
        teacher_out = teacher_out.detach()
        teacher_prob = F.softmax(
            (teacher_out - self.center) / teacher_temp, dim=-1
        )
        loss = torch.sum(-teacher_prob * student_logp, dim=-1).mean()
        return loss

    def forward(
        self,
        x1: torch.Tensor | None = None,
        x2: torch.Tensor | None = None,
        **kwargs,
    ):
        if x1 is None or x2 is None:
            x1 = kwargs.get("x1")
            x2 = kwargs.get("x2")
        assert x1 is not None and x2 is not None, "ECGDINO requires x1 and x2"

        student_feat = self.student_backbone(x1)
        student_out = self.student_head(student_feat)

        with torch.no_grad():
            teacher_feat = self.teacher_backbone(x2)
            teacher_out = self.teacher_head(teacher_feat)

        loss = self.dino_loss(student_out, teacher_out)

        with torch.no_grad():
            self.update_center(teacher_out.detach())

        return {"loss": loss, "logits": student_out}


class ECGDINODataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        if torch.rand(1).item() < 0.5:
            x = x * (0.9 + 0.2 * torch.rand(1).item())
        if torch.rand(1).item() < 0.5:
            x = x + 0.01 * torch.randn_like(x)
        if torch.rand(1).item() < 0.5:
            shift = int(torch.randint(-5, 6, (1,)).item())
            x = torch.roll(x, shifts=shift, dims=-1)
        return x

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # [1, L]
        x1 = self.augment(x)
        x2 = self.augment(x)
        return {"x1": x1, "x2": x2}


def dino_collator(batch):
    x1 = torch.stack([b["x1"] for b in batch])
    x2 = torch.stack([b["x2"] for b in batch])
    return {"x1": x1, "x2": x2}


class DINOTeacherUpdateCallback(TrainerCallback):
    def __init__(self, momentum: float = 0.996):
        self.momentum = momentum

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            model.update_teacher(momentum=self.momentum)


class DINOFeatureStdCallback(TrainerCallback):
    """Every `interval` epochs, compute mean feature std over first n_samples train embeddings."""

    def __init__(self, X_train: np.ndarray, n_samples: int = 2000, interval: int = 10):
        self.X_train = X_train
        self.n_samples = min(n_samples, len(X_train))
        self.interval = interval

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or int(state.epoch) % self.interval != 0:
            return
        backbone = model.student_backbone.eval()
        device = next(backbone.parameters()).device
        embeddings = []
        with torch.no_grad():
            for i in range(self.n_samples):
                x = (
                    torch.tensor(self.X_train[i], dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                z = backbone(x)
                embeddings.append(z)
        embeddings = torch.cat(embeddings)
        mean_std = embeddings.std(dim=0).mean().item()
        print(f"Epoch {int(state.epoch)} — Mean feature std: {mean_std:.4f}")


def build_classifier_from_dino(
    dino_model: ECGDINO,
    n_classes: int = 5,
    class_weights: torch.Tensor | None = None,
):
    """Build a classification model from the DINO student backbone (same interface as MAE)."""
    from mae import TinyTransformerMAEFinetune

    backbone = dino_model.student_backbone
    n_layer = len(backbone.encoder.layers)

    model = TinyTransformerMAEFinetune(
        n_classes=n_classes,
        patch_size=backbone.patch_size,
        seq_len=backbone.seq_len,
        embed_dim=backbone.embed_dim,
        n_layer=n_layer,
        class_weights=class_weights,
    )
    model.patch_embed.load_state_dict(backbone.patch_embed.state_dict())
    model.pos.data.copy_(backbone.pos.data)
    model.encoder.load_state_dict(backbone.encoder.state_dict())
    model.final_norm.load_state_dict(backbone.norm.state_dict())
    return model


def main():
    parser = argparse.ArgumentParser()
    add_common_ecg_cli_args(parser, output_dir_default="./tiny_ecg_dino_runs")
    parser.add_argument("--teacher_momentum", type=float, default=0.996)
    parser.add_argument("--seq_len", type=int, default=198)
    parser.add_argument("--patch_size", type=int, default=9)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--out_dim", type=int, default=256)
    args = parser.parse_args()

    seed_everything(SEED)

    X, RR, y = extract_beats_and_rr(args.folder, denoise=True)
    X = normalize_rows(X)

    print(f"Loaded beats: {len(y)}")
    class_counts = {IDX2CLS[i]: int((y == i).sum()) for i in range(5)}
    print("Class counts:", class_counts)

    X_train, X_tmp, RR_train, RR_tmp, y_train, y_tmp = train_test_split(
        X, RR, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_valid, X_test, RR_valid, RR_test, y_valid, y_test = train_test_split(
        X_tmp, RR_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=SEED
    )

    if args.use_noise_aug:
        X_train = maybe_augment_noise(X_train, args.nstdb_folder, args.snr_db)

    dino_train_dataset = ECGDINODataset(X_train)
    dino_valid_dataset = ECGDINODataset(X_valid)

    print("\n=== Stage 1: DINO pretraining ===")
    dino_model = ECGDINO(
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        n_layer=args.n_layer,
        out_dim=args.out_dim,
    )

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

        dino_trainer = Trainer(
            model=dino_model,
            args=pretrain_args,
            train_dataset=dino_train_dataset,
            eval_dataset=dino_valid_dataset,
            data_collator=dino_collator,
            callbacks=[
                DINOTeacherUpdateCallback(momentum=args.teacher_momentum),
                DINOFeatureStdCallback(X_train=X_train, n_samples=2000, interval=10),
            ],
        )
        dino_trainer.train()
        print("DINO validation:", dino_trainer.evaluate())
    else:
        from safetensors.torch import load_file

        state_dict = load_file(args.checkpoint)
        dino_model.load_state_dict(state_dict)

    print("\n=== Stage 2: classifier finetuning ===")

    X_train_ft, y_train_ft, RR_train_ft = percent_trained(
        X_train, y_train, RR_train, args
    )
    class_counts_ft = np.bincount(y_train_ft, minlength=5).astype(np.float32)
    class_weights_ft_np = class_counts_ft.sum() / (
        len(class_counts_ft) * class_counts_ft + 1e-8
    )
    class_weights_ft = torch.tensor(class_weights_ft_np, dtype=torch.float32)

    clf_model = build_classifier_from_dino(
        dino_model,
        n_classes=len(class_counts_ft),
        class_weights=class_weights_ft if args.balanced_weight else None,
    )
    train_dataset = ECGRRDataset(X_train_ft, RR_train_ft, y_train_ft)
    valid_dataset = ECGRRDataset(X_valid, RR_valid, y_valid)
    test_dataset = ECGRRDataset(X_test, RR_test, y_test)

    finetune_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "finetune"),
        epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        lr=args.finetune_lr,
        seed=SEED,
    )
    finetune_args.metric_for_best_model = "macro_f1"
    finetune_args.greater_is_better = True

    clf_trainer = Trainer(
        model=clf_model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=cls_collator,
        compute_metrics=compute_metrics,
    )
    clf_trainer.train()

    print("Validation metrics:", clf_trainer.evaluate())
    print("Test metrics:", clf_trainer.evaluate(eval_dataset=test_dataset))

    pred_output = clf_trainer.predict(test_dataset)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=[IDX2CLS[i] for i in range(5)],
            zero_division=0,
        )
    )
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]))


if __name__ == "__main__":
    main()
