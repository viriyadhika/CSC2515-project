#!/usr/bin/env python3
from __future__ import annotations

"""
Common, project-wide ECG utilities and constants.

This module holds everything that is shared between:
- novel models (MAE variants, DINO, etc.)
- paper reproduction scripts
"""

import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import wfdb
from scipy.signal import butter, filtfilt, medfilt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import TrainingArguments

# ---------------------------------------------------------------------------
# Constants / label mapping
# ---------------------------------------------------------------------------

SEED = 42
FS = 360
WINDOW = 99  # 198 total samples centered on R peak
EXCLUDED_RECORDS = {"102", "104", "107", "217"}

AAMI_MAP = {
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    "V": 2,
    "E": 2,
    "F": 3,
    "/": 4,
    "f": 4,
    "Q": 4,
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
    k1 = int(0.2 * fs)
    k2 = int(0.6 * fs)
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
    noise = noise_stream[start : start + len(window)].copy()

    ps = np.mean(window**2)
    pn = np.mean(noise**2) + 1e-8
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
        return np.asarray(
            [add_em_noise(x, noise_stream, snr_db, rng) for x in X_train],
            dtype=np.float32,
        )

    noise = rng.normal(0.0, 0.03, size=X_train.shape)
    return (X_train + noise).astype(np.float32)


def extract_beats_and_rr(
    folder: str, denoise: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Datasets / collators
# ---------------------------------------------------------------------------


class ECGRRDataset(Dataset):
    def __init__(self, X: np.ndarray, rr: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.rr = torch.tensor(rr, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx].unsqueeze(0),
            "rr": self.rr[idx],
            "labels": self.y[idx],
        }


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
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


def make_training_args(
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    save_strategy: str = "epoch",
) -> TrainingArguments:
    kwargs = dict(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy=save_strategy,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    try:
        return TrainingArguments(eval_strategy="epoch", **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **kwargs)


__all__ = [
    "SEED",
    "FS",
    "WINDOW",
    "EXCLUDED_RECORDS",
    "AAMI_MAP",
    "IDX2CLS",
    "seed_everything",
    "normalize_rows",
    "baseline_remove_and_lowpass",
    "load_electrode_motion_noise",
    "add_em_noise",
    "maybe_augment_noise",
    "extract_beats_and_rr",
    "ECGRRDataset",
    "compute_metrics",
    "percent_trained",
    "make_training_args",
]

