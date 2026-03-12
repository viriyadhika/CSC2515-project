#!/usr/bin/env python3
"""
Reproduction-oriented script for:
"Classification of cardiac arrhythmia using a convolutional neural network and
bi-directional long short-term memory" (PMC9152186).

Paper-aligned choices implemented here:
- MIT-BIH, 5 classes (N/S/V/F/Q)
- 3 s before + 3 s after the beat => 2161 samples at 360 Hz
- 60% / 20% / 20% patient-level split
- balance via simple over/under-sampling, not SMOTE
- CNN(32, k=5) x2 + MaxPool + BiLSTM(32) x2 + Dense16 + Dense5
- Adam, 100 epochs by default, weight decay as L2 approximation

This is still a best-effort code reproduction because the paper's exact patient split
is random, not published as a fixed list.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import wfdb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SEED = 42
FS = 360
SEC_BEFORE = 3
SEC_AFTER = 3
WINDOW_LEFT = SEC_BEFORE * FS
WINDOW_RIGHT = SEC_AFTER * FS
INPUT_LEN = WINDOW_LEFT + WINDOW_RIGHT + 1


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_label(symbol: str):
    return AAMI_MAP.get(symbol)


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < eps, eps, std)
    return (x - mean) / std


def extract_beat_windows(records: List[str], folder: str) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for rec in records:
        path = os.path.join(folder, rec)
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")
        sig = record.p_signal[:, 0]
        for peak, sym in zip(ann.sample, ann.symbol):
            encoded = encode_label(sym)
            if encoded is None:
                continue
            start = peak - WINDOW_LEFT
            end = peak + WINDOW_RIGHT + 1
            if start < 0 or end > len(sig):
                continue
            X.append(sig[start:end])
            y.append(encoded)
    return np.asarray(X), np.asarray(y)


def rebalance_resample(X: np.ndarray, y: np.ndarray, random_state: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    indices_by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    target = int(np.median([len(v) for v in indices_by_class.values()]))
    chosen = []
    for c, idx in indices_by_class.items():
        if len(idx) >= target:
            sampled = rng.choice(idx, size=target, replace=False)
        else:
            sampled = rng.choice(idx, size=target, replace=True)
        chosen.append(sampled)
    final_idx = np.concatenate(chosen)
    rng.shuffle(final_idx)
    return X[final_idx], y[final_idx]


class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]


class CNNBiLSTM9152186(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.conv2 = nn.Conv1d(32, 32, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=32, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="mit-bih-arrhythmia-database-1.0.0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    args = parser.parse_args()

    seed_everything(SEED)
    records = sorted(f[:-4] for f in os.listdir(args.folder) if f.endswith(".hea"))

    train_records, temp_records = train_test_split(records, test_size=0.4, random_state=SEED)
    valid_records, test_records = train_test_split(temp_records, test_size=0.5, random_state=SEED)

    X_train, y_train = extract_beat_windows(train_records, args.folder)
    X_valid, y_valid = extract_beat_windows(valid_records, args.folder)
    X_test, y_test = extract_beat_windows(test_records, args.folder)

    X_train = normalize_rows(X_train)
    X_valid = normalize_rows(X_valid)
    X_test = normalize_rows(X_test)

    X_train, y_train = rebalance_resample(X_train, y_train, random_state=SEED)

    print("Train class counts:", Counter(y_train))
    print("Valid class counts:", Counter(y_valid))
    print("Test class counts:", Counter(y_test))

    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(ECGDataset(X_valid, y_valid), batch_size=args.batch_size)
    test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNBiLSTM9152186().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state, best_val_f1 = None, -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, valid_loader, device)
        print(
            f"epoch={epoch:03d} loss={tr_loss:.4f} "
            f"val_acc={val['acc']:.4f} val_bal_acc={val['bal_acc']:.4f} val_macro_f1={val['macro_f1']:.4f}"
        )
        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, test_loader, device)
    print(f"TEST acc={test['acc']:.4f} bal_acc={test['bal_acc']:.4f} macro_f1={test['macro_f1']:.4f}")
    print(classification_report(
        test["y_true"], test["y_pred"], labels=[0, 1, 2, 3, 4],
        target_names=[IDX2CLS[i] for i in range(5)], zero_division=0
    ))
    print(confusion_matrix(test["y_true"], test["y_pred"], labels=[0, 1, 2, 3, 4]))


if __name__ == "__main__":
    main()
