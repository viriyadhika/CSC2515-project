#!/usr/bin/env python3
"""
JEPA-style ECG pretraining + finetuning script.

This mirrors the overall pipeline of `mae.py` (same data preprocessing,
train/valid/test splits, and finetuning classifier) but replaces the MAE
reconstruction objective with a Joint Embedding Predictive Architecture
style objective in latent space.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

from common.lib import (
    SEED,
    FS,
    WINDOW,
    IDX2CLS,
    seed_everything,
    maybe_augment_noise,
    extract_beats_and_rr,
    preprocess_beats,
    balance_classes,
    ECGRRDataset,
    compute_metrics,
    percent_trained,
    make_training_args,
)
from novel.mae_lib import (
    ECGMAEDataset,
    mae_collator,
    cls_collator,
    add_common_ecg_cli_args,
    evaluate_knn_and_tsne_on_test,
)


class ECGJEPA(nn.Module):
    """
    Simple JEPA-style encoder that predicts latent representations of
    masked tokens from their own latent representations, in the spirit
    of joint embedding predictive architectures.
    """

    def __init__(
        self,
        seq_len: int = 198,
        patch_size: int = 9,
        embed_dim: int = 64,
        nhead: int = 8,
        n_layer: int = 4,
        mask_ratio: float = 0.6,
        pred_hidden_dim: int = 128,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim

        # Patch embedding (same tokenizer style as MAE)
        self.patch_embed = nn.Conv1d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Predictor operates in latent space
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, pred_hidden_dim),
            nn.GELU(),
            nn.Linear(pred_hidden_dim, embed_dim),
        )

        self.loss_fn = nn.MSELoss()

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, L]
        tokens = self.patch_embed(x).transpose(1, 2)  # [B, N, D]
        tokens = tokens + self.pos[:, : tokens.size(1), :]
        return tokens

    def _random_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        returns a boolean mask of shape [B, N] indicating which tokens
        to predict (masked positions).
        """
        B, N, _ = x.shape
        num_mask = int(N * self.mask_ratio)

        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        for b in range(B):
            idx = torch.randperm(N, device=x.device)[:num_mask]
            mask[b, idx] = True
        return mask

    def forward(self, x: torch.Tensor | None = None, labels=None):
        """
        Forward used by HuggingFace Trainer.
        x : [B, 1, L]

        Returns:
            dict with keys "loss" and "logits".
        """
        if x is None:
            raise ValueError("ECGJEPA.forward expects input tensor x")

        tokens = self._tokenize(x)  # [B, N, D]

        latent = self.encoder(tokens)
        latent = self.encoder_norm(latent)  # [B, N, D]

        # Predict in latent space for a subset of tokens
        mask = self._random_mask(latent)  # [B, N]

        target_latent = latent.detach()
        pred_latent = self.predictor(latent)

        # Select masked positions only
        target_masked = target_latent[mask]  # [B * M, D]
        pred_masked = pred_latent[mask]      # [B * M, D]

        loss = self.loss_fn(pred_masked, target_masked)

        # For logging, we can expose the predicted latents
        return {"loss": loss, "logits": pred_masked}

    def build_classifier(
        self,
        n_classes: int = 5,
        class_weights: torch.Tensor | None = None,
    ):
        """
        Build a classifier initialized from the JEPA encoder, mirroring
        the MAE finetuning classifier structure.
        """
        model = TinyTransformerJEPAFinetune(
            n_classes=n_classes,
            patch_size=self.patch_size,
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
            n_layer=len(self.encoder.layers),
            class_weights=class_weights,
        )

        # Initialize from JEPA encoder
        model.patch_embed.load_state_dict(self.patch_embed.state_dict())
        model.pos.data.copy_(self.pos.data)
        model.encoder.load_state_dict(self.encoder.state_dict())
        model.final_norm.load_state_dict(self.encoder_norm.state_dict())

        return model


class TinyTransformerJEPAFinetune(nn.Module):
    """
    Transformer classifier initialized from `ECGJEPA` encoder.
    Identical to the MAE finetuning classifier but without RR inputs.
    """

    def __init__(
        self,
        n_layer: int = 1,
        embed_dim: int = 128,
        n_classes: int = 5,
        patch_size: int = 9,
        seq_len: int = 198,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        num_patches = seq_len // patch_size

        self.patch_embed = nn.Conv1d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.2,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layer)
        self.final_norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, n_classes)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x=None, labels=None):
        x_tokens = self.patch_embed(x).transpose(1, 2)
        x_tokens = x_tokens + self.pos[:, : x_tokens.size(1), :]

        x_tokens = self.encoder(x_tokens)
        x_tokens = self.final_norm(x_tokens)

        x_pooled = x_tokens.mean(dim=1)
        logits = self.head(x_pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def jepa_pretrain_from_datasets(
    train_dataset,
    valid_dataset,
    jepa_model: nn.Module,
    training_args: TrainingArguments,
):
    """
    Generic JEPA pretraining that only depends on datasets.

    Each dataset item is expected to include a key "x" with a tensor
    shaped for the JEPA encoder (e.g., [1, L] for 1D signals).
    """
    trainer = Trainer(
        model=jepa_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=mae_collator,
    )
    trainer.train()
    return trainer


def jepa_finetune_from_datasets(
    clf_model: nn.Module,
    train_dataset,
    valid_dataset,
    test_dataset,
    training_args: TrainingArguments,
    data_collator=cls_collator,
    compute_metrics_fn=compute_metrics,
):
    """
    Generic classifier finetuning for a JEPA-initialized backbone.

    Datasets are expected to return a dict with keys "x" and "labels".
    """
    clf_trainer = Trainer(
        model=clf_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    clf_trainer.train()

    val_metrics = clf_trainer.evaluate()
    test_metrics = clf_trainer.evaluate(eval_dataset=test_dataset)
    return clf_trainer, val_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser()
    add_common_ecg_cli_args(parser, output_dir_default="./data/tiny_ecg_jepa_runs")

    parser.add_argument("--mask_ratio", type=float, default=0.6)
    args = parser.parse_args()

    seed_everything(SEED)

    # Shared beat + RR extraction, then JEPA-specific preprocessing
    X, RR, y = extract_beats_and_rr(args.folder, pre_process=None)
    X = preprocess_beats(X)

    print(f"Loaded beats: {len(y)}")
    class_counts = {IDX2CLS[i]: int((y == i).sum()) for i in range(5)}
    print("Class counts:", class_counts)

    # 7:1:2 split (same as TinyTransformer / MAE / DINO scripts)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=SEED
    )

    if args.use_noise_aug:
        X_train = maybe_augment_noise(X_train, args.nstdb_folder, args.snr_db)

    jepa_train_dataset = ECGMAEDataset(X_train)
    jepa_valid_dataset = ECGMAEDataset(X_valid)

    print("\n=== Stage 1: JEPA pretraining ===")
    jepa_model = ECGJEPA(mask_ratio=args.mask_ratio)

    pretrain_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "jepa_pretrain"),
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        lr=args.pretrain_lr,
        seed=SEED,
    )
    pretrain_args.metric_for_best_model = "eval_loss"
    pretrain_args.greater_is_better = False

    if args.checkpoint is None:
        jepa_trainer = jepa_pretrain_from_datasets(
            train_dataset=jepa_train_dataset,
            valid_dataset=jepa_valid_dataset,
            jepa_model=jepa_model,
            training_args=pretrain_args,
        )
        print("JEPA validation:")
        print(jepa_trainer.evaluate())
    else:
        from safetensors.torch import load_file

        state_dict = load_file(args.checkpoint)
        jepa_model.load_state_dict(state_dict)

    # === Zero-shot KNN + t-SNE on test set ===
    def _jepa_get_embeddings(X_np: np.ndarray) -> np.ndarray:
        jepa_model.eval()
        with torch.no_grad():
            device = next(jepa_model.parameters()).device
            X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(1)
            tokens = jepa_model._tokenize(X_tensor)
            latent = jepa_model.encoder(tokens)
            latent = jepa_model.encoder_norm(latent)
            return latent.mean(dim=1).cpu().numpy()

    evaluate_knn_and_tsne_on_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        get_embeddings=_jepa_get_embeddings,
        idx2cls=IDX2CLS,
        output_dir=args.output_dir,
        prefix="jepa",
    )

    print("\n=== Stage 2: classifier finetuning ===")

    # Optionally subsample labeled training data
    X_train, y_train = percent_trained(X_train, y_train, args)

    # Rebalance training set only
    X_train, y_train = balance_classes(
        X_train,
        y_train,
        target_size=5000,
        seed=SEED,
        n_classes=5,
    )

    class_counts = np.bincount(y_train, minlength=5).astype(np.float32)
    class_weights_np = class_counts.sum() / (len(class_counts) * class_counts + 1e-8)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    clf_model = jepa_model.build_classifier(
        n_classes=len(class_counts),
        class_weights=class_weights if args.balanced_weight else None,
    )

    train_dataset = ECGRRDataset(X_train, y_train)
    valid_dataset = ECGRRDataset(X_valid, y_valid)
    test_dataset = ECGRRDataset(X_test, y_test)

    finetune_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "finetune"),
        epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        lr=args.finetune_lr,
        seed=SEED,
    )
    finetune_args.metric_for_best_model = "macro_f1"
    finetune_args.greater_is_better = True

    clf_trainer, val_metrics, test_metrics = jepa_finetune_from_datasets(
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

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=[IDX2CLS[i] for i in range(5)],
            zero_division=0,
        )
    )
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
        )
    )


if __name__ == "__main__":
    main()

