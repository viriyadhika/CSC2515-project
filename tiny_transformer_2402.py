#!/usr/bin/env python3
"""
Hugging Face Trainer reproduction script for:

"A Tiny Transformer for Low-Power Arrhythmia Classification on Microcontrollers"
(arXiv:2402.10748 / IEEE TBioCAS 2024)

Paper-aligned choices kept:
- MIT-BIH 5-class AAMI setup (N/S/V/F/Q)
- exclude paced-beat records 102, 104, 107, 217
- heartbeat window length 198 samples around R peak
- additional RR interval input (pre-RR and post-RR)
- conv embedding with k=31, stride=3, embedding dim=16
- 1 transformer encoder block, 8 heads, FF dim=128
- 7:1:2 random train/valid/test split for intra-patient evaluation
- optional electrode-motion noise augmentation using NSTDB if available

Notes:
- Single-split and K-fold evaluation are both implemented with Hugging Face Trainer.
- Validation is used for model selection; test is evaluated only after training.
- This reports full-precision PyTorch metrics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from common.dataloader import ECGLoader
from common.lib import (
    SEED,
    FS,
    WINDOW,
    EXCLUDED_RECORDS,
    AAMI_MAP,
    IDX2CLS,
    seed_everything,
    normalize_rows,
    baseline_remove_and_lowpass,
    low_pass_filter,
    load_electrode_motion_noise,
    add_em_noise,
    maybe_augment_noise,
    extract_beats_and_rr,
    ECGRRDataset,
    compute_metrics,
    percent_trained,
    balance_classes,
)


class TinyTransformer2402(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()

        # For input length 198, Conv1d(kernel=31, stride=3, padding=0) => 56 tokens
        self.embed = nn.Conv1d(1, 16, kernel_size=3, stride=3)
        self.pos = nn.Parameter(torch.zeros(1, 66, 16))

        enc = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=8,
            dim_feedforward=128,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.2,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)

        self.final_norm = nn.LayerNorm(16)
        self.rr_proj = nn.Linear(2, 2)
        self.head = nn.Linear(16, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x=None, rr=None, labels=None):
        x = self.embed(x).transpose(1, 2)              # [B, 56, 16]
        x = x + self.pos[:, :x.size(1), :]
        x = self.encoder(x)
        x = self.final_norm(x)
        x = x.mean(dim=1)                             # [B, 16]

        # rr = self.rr_proj(rr)                         # [B, 2]
        # x = torch.cat([x, rr], dim=1)                 # [B, 17]

        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def data_collator(features):
    x = torch.stack([f["x"] for f in features])
    # rr = torch.stack(["f["rr""] for f in features])
    labels = torch.tensor([int(f["labels"]) for f in features], dtype=torch.long)
    return {"x": x, "labels": labels}


def make_training_args(
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> TrainingArguments:
    kwargs = dict(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Compatibility across transformers versions:
    try:
        return TrainingArguments(eval_strategy="epoch", **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **kwargs)


def train_and_test_with_trainer(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    args,
    run_name: str,
):
    model = TinyTransformer2402()

    training_args = make_training_args(
        output_dir=str(Path(args.output_dir) / run_name),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)

    pred_output = trainer.predict(test_dataset)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids

    return {
        "trainer": trainer,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def percent_trained(X_train, y_train, args):
    if args.percent_train < 100.0:
        frac = args.percent_train / 100.0
    
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=frac,
            stratify=y_train,
            random_state=SEED,
        )
    
        print(f"Using {len(y_train)} labeled training beats ({args.percent_train}%)")

        return X_train, y_train
    return X_train, y_train

def run_single_split(X: np.ndarray, RR: np.ndarray, y: np.ndarray, args) -> None:
    # Paper-style 7:1:2 random intra-patient split
    X_train, X_tmp, RR_train, RR_tmp, y_train, y_tmp = train_test_split(
        X, RR, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_valid, X_test, RR_valid, RR_test, y_valid, y_test = train_test_split(
        X_tmp, RR_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=SEED
    )

    if args.use_noise_aug:
        X_train = maybe_augment_noise(X_train, args.nstdb_folder, args.snr_db)

    # Subsample labeled training data if requested
    X_train, y_train = percent_trained(X_train, y_train, args)

    # Rebalance training set only (valid/test keep real distribution)
    X_train, y_train = balance_classes(
        X_train,
        y_train,
        target_size=5000,
        seed=SEED,
        n_classes=5,
    )

    train_dataset = ECGRRDataset(X_train, y_train)
    valid_dataset = ECGRRDataset(X_valid, y_valid)
    test_dataset = ECGRRDataset(X_test, y_test)

    result = train_and_test_with_trainer(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        args=args,
        run_name="single_split",
    )

    print("Validation metrics:")
    print(result["val_metrics"])

    print("Test metrics:")
    print(result["test_metrics"])

    print(classification_report(
        result["y_true"],
        result["y_pred"],
        labels=[0, 1, 2, 3, 4],
        target_names=[IDX2CLS[i] for i in range(5)],
        zero_division=0,
    ))
    print(confusion_matrix(
        result["y_true"],
        result["y_pred"],
        labels=[0, 1, 2, 3, 4],
    ))


def run_kfold(X: np.ndarray, RR: np.ndarray, y: np.ndarray, args) -> None:
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train_all, X_test = X[train_idx], X[test_idx]
        RR_train_all, RR_test = RR[train_idx], RR[test_idx]
        y_train_all, y_test = y[train_idx], y[test_idx]

        # Keep the same 7:1:2 logic within each fold:
        # first CV train portion is split into train/valid = 7:1 approximately via 1/8 valid split
        X_train, X_valid, RR_train, RR_valid, y_train, y_valid = train_test_split(
            X_train_all,
            RR_train_all,
            y_train_all,
            test_size=1 / 8,
            stratify=y_train_all,
            random_state=SEED,
        )

        if args.use_noise_aug:
            X_train = maybe_augment_noise(X_train, args.nstdb_folder, args.snr_db)

        # Subsample labeled training data if requested
        X_train, y_train = percent_trained(X_train, y_train, args)

        # Rebalance training set only (valid/test keep real distribution)
        X_train, y_train = balance_classes(
            X_train,
            y_train,
            target_size=5000,
            seed=SEED,
            n_classes=5,
        )

        train_dataset = ECGRRDataset(X_train, y_train)
        valid_dataset = ECGRRDataset(X_valid, y_valid)
        test_dataset = ECGRRDataset(X_test, y_test)

        print(f"\n=== Fold {fold}/{args.folds} ===")

        result = train_and_test_with_trainer(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            args=args,
            run_name=f"fold_{fold}",
        )

        tm = result["test_metrics"]
        fold_metrics.append({
            "accuracy": float(tm["eval_accuracy"]),
            "balanced_accuracy": float(tm["eval_balanced_accuracy"]),
            "macro_f1": float(tm["eval_macro_f1"]),
        })

        print(f"fold={fold} test_acc={tm['eval_accuracy']:.4f} "
              f"test_bal_acc={tm['eval_balanced_accuracy']:.4f} "
              f"test_macro_f1={tm['eval_macro_f1']:.4f}")

    print("\n=== Cross-validation summary ===")
    print("Mean accuracy:", np.mean([m["accuracy"] for m in fold_metrics]))
    print("Mean balanced accuracy:", np.mean([m["balanced_accuracy"] for m in fold_metrics]))
    print("Mean macro F1:", np.mean([m["macro_f1"] for m in fold_metrics]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data/mit-bih-arrhythmia-database-1.0.0")
    parser.add_argument("--nstdb_folder", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_noise_aug", action="store_true")
    parser.add_argument("--snr_db", type=float, default=12.0)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./tiny_ecg_runs")
    parser.add_argument(
        "--percent_train",
        type=float,
        default=100.0,
        help="Percent of labeled training data to use for finetuning (0-100)"
    )
    args = parser.parse_args()

    seed_everything(SEED)

    full_data = ECGLoader(
        args,
        pre_process=low_pass_filter,
        post_process=normalize_rows,
        include_rr=True,
    ).load_full()
    X = full_data["X"]
    RR = full_data["RR"]
    y = full_data["y"]

    if args.folds > 1:
        run_kfold(X, RR, y, args)
    else:
        run_single_split(X, RR, y, args)


if __name__ == "__main__":
    main()
