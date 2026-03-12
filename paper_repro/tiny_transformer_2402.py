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
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import wfdb
from scipy.signal import butter, filtfilt, medfilt
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


SEED = 42
FS = 360
WINDOW = 99  # 198 total samples centered on R peak

EXCLUDED_RECORDS = {'102', '104', '107', '217'}
AAMI_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4,
}
IDX2CLS = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def encode_label(symbol: str):
    return AAMI_MAP.get(symbol)


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < eps, eps, std)
    return (x - mean) / std


def baseline_remove_and_lowpass(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    # Approximate the paper's 200 ms and 600 ms median baseline removal + 35 Hz low-pass.
    k1 = int(0.2 * fs)
    k2 = int(0.6 * fs)

    # Make odd kernel sizes for medfilt
    if k1 % 2 == 0:
        k1 += 1
    if k2 % 2 == 0:
        k2 += 1

    baseline = medfilt(sig, kernel_size=k1)
    baseline = medfilt(baseline, kernel_size=k2)
    sig = sig - baseline

    b, a = butter(4, 35 / (fs / 2), btype="low")
    return filtfilt(b, a, sig)


def load_electrode_motion_noise(nstdb_folder: str) -> np.ndarray:
    # Best effort: concatenate lead-0 from all available NSTDB records.
    records = sorted(f[:-4] for f in os.listdir(nstdb_folder) if f.endswith(".hea"))
    if not records:
        raise RuntimeError(f"No NSTDB .hea records found in: {nstdb_folder}")

    chunks = []
    for rec in records:
        record = wfdb.rdrecord(os.path.join(nstdb_folder, rec))
        chunks.append(record.p_signal[:, 0])
    return np.concatenate(chunks)


def add_em_noise(
    window: np.ndarray,
    noise_stream: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(noise_stream) < len(window):
        return window

    start = int(rng.integers(0, len(noise_stream) - len(window) + 1))
    noise = noise_stream[start:start + len(window)].copy()

    ps = np.mean(window ** 2)
    pn = np.mean(noise ** 2) + 1e-8
    scale = np.sqrt(ps / (pn * (10 ** (snr_db / 10))))
    return window + scale * noise


def maybe_augment_noise(
    X_train: np.ndarray,
    nstdb_folder: str | None,
    snr_db: float,
) -> np.ndarray:
    rng = np.random.default_rng(SEED)

    if nstdb_folder and os.path.isdir(nstdb_folder):
        noise_stream = load_electrode_motion_noise(nstdb_folder)
        return np.asarray([add_em_noise(x, noise_stream, snr_db, rng) for x in X_train])

    # Fallback if NSTDB is unavailable.
    noise = rng.normal(0.0, 0.03, size=X_train.shape)
    return X_train + noise


def extract_beats_and_rr(folder: str, denoise: bool = True):
    X, RR, y = [], [], []
    records = sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".hea"))

    for rec in records:
        if rec in EXCLUDED_RECORDS:
            continue

        path = os.path.join(folder, rec)
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")

        sig = record.p_signal[:, 0]
        if denoise:
            sig = baseline_remove_and_lowpass(sig)

        peaks = ann.sample
        syms = ann.symbol

        for i, (peak, sym) in enumerate(zip(peaks, syms)):
            lab = encode_label(sym)
            if lab is None:
                continue

            if i == 0 or i == len(peaks) - 1:
                continue

            start = peak - WINDOW
            end = peak + WINDOW
            if start < 0 or end > len(sig):
                continue

            pre_rr = peaks[i] - peaks[i - 1]
            post_rr = peaks[i + 1] - peaks[i]

            beat = sig[start:end]
            if len(beat) != 2 * WINDOW:
                continue

            X.append(beat)
            RR.append([pre_rr, post_rr])
            y.append(lab)

    X = np.asarray(X, dtype=np.float32)
    RR = np.asarray(RR, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    RR = np.clip((RR / FS - 1.0), -2.0, 2.0)
    return X, RR, y


class ECGRRDataset(Dataset):
    def __init__(self, X: np.ndarray, rr: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.rr = torch.tensor(rr, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx].unsqueeze(0),  # [1, 198]
            "rr": self.rr[idx],             # [2]
            "labels": self.y[idx],
        }


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
        self.head = nn.Linear(18, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x=None, rr=None, labels=None):
        x = self.embed(x).transpose(1, 2)              # [B, 56, 16]
        x = x + self.pos[:, :x.size(1), :]
        x = self.encoder(x)
        x = self.final_norm(x)
        x = x.mean(dim=1)                             # [B, 16]

        rr = self.rr_proj(rr)                         # [B, 2]
        x = torch.cat([x, rr], dim=1)                 # [B, 17]

        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def data_collator(features):
    x = torch.stack([f["x"] for f in features])
    rr = torch.stack([f["rr"] for f in features])
    labels = torch.tensor([int(f["labels"]) for f in features], dtype=torch.long)
    return {"x": x, "rr": rr, "labels": labels}


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


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


def percent_trained(X_train, y_train, RR_train, args):
    if args.percent_train < 100.0:
        frac = args.percent_train / 100.0
    
        X_train, _, RR_train, _, y_train, _ = train_test_split(
            X_train,
            RR_train,
            y_train,
            train_size=frac,
            stratify=y_train,
            random_state=SEED,
        )
    
        print(f"Using {len(y_train)} labeled training beats ({args.percent_train}%)")

        return X_train, y_train, RR_train
    return X_train, y_train, RR_train

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

    X_train, y_train, RR_train = percent_trained(X_train, y_train, RR_train, args)

    train_dataset = ECGRRDataset(X_train, RR_train, y_train)
    valid_dataset = ECGRRDataset(X_valid, RR_valid, y_valid)
    test_dataset = ECGRRDataset(X_test, RR_test, y_test)

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

        X_train, y_train, RR_train = percent_trained(X_train, y_train, RR_train, args)

        train_dataset = ECGRRDataset(X_train, RR_train, y_train)
        valid_dataset = ECGRRDataset(X_valid, RR_valid, y_valid)
        test_dataset = ECGRRDataset(X_test, RR_test, y_test)

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
    parser.add_argument("--folder", type=str, default="mit-bih-arrhythmia-database-1.0.0")
    parser.add_argument("--nstdb_folder", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
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

    X, RR, y = extract_beats_and_rr(args.folder, denoise=True)
    X = normalize_rows(X)

    print(f"Loaded beats: {len(y)}")
    class_counts = {IDX2CLS[i]: int((y == i).sum()) for i in range(5)}
    print("Class counts:", class_counts)

    if args.folds > 1:
        run_kfold(X, RR, y, args)
    else:
        run_single_split(X, RR, y, args)


if __name__ == "__main__":
    main()