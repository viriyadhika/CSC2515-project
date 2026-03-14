#!/usr/bin/env python3
from __future__ import annotations

"""
MAE-specific datasets and collators used by the novel models.
"""

from typing import List, Dict, Callable
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from common.lib import SEED, make_training_args  # re-exported for convenience


class ECGMAEDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx].unsqueeze(0)}


def mae_collator(features: List[Dict]):
    x = torch.stack([f["x"] for f in features])
    labels = torch.stack([f["x"] for f in features])
    return {"x": x, "labels": labels}


def cls_collator(features: List[Dict]):
    x = torch.stack([f["x"] for f in features])
    labels = torch.tensor([int(f["labels"]) for f in features], dtype=torch.long)
    return {"x": x, "labels": labels}


def add_common_ecg_cli_args(
    parser: argparse.ArgumentParser,
    *,
    output_dir_default: str,
) -> argparse.ArgumentParser:
    """
    Add common CLI args shared by `mae.py`, `mae_freq.py`, and `dino.py`.

    This covers dataset paths, train/finetune hyperparameters, augmentation flags,
    and output/checkpoint options. Model-specific args should remain in each script.
    """
    parser.add_argument(
        "--folder", type=str, default="data/mit-bih-arrhythmia-database-1.0.0"
    )
    parser.add_argument("--nstdb_folder", type=str, default=None)

    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--finetune_epochs", type=int, default=40)

    parser.add_argument("--pretrain_batch_size", type=int, default=256)
    parser.add_argument("--finetune_batch_size", type=int, default=128)

    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-3)

    parser.add_argument("--use_noise_aug", action="store_true")
    parser.add_argument("--balanced_weight", action="store_false")
    parser.add_argument("--snr_db", type=float, default=12.0)

    parser.add_argument("--output_dir", type=str, default=output_dir_default)
    parser.add_argument(
        "--percent_train",
        type=float,
        default=100.0,
        help="Percent of labeled training data to use for finetuning (0-100)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)

    return parser


def evaluate_knn_and_tsne_on_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    get_embeddings: Callable[[np.ndarray], np.ndarray],
    idx2cls: Dict[int, str],
    output_dir: str,
    prefix: str,
) -> None:
    """
    Evaluate zero-shot KNN in raw vs encoder space and visualize t-SNE
    on the encoder embeddings of the test set.

    Parameters
    ----------
    X_train, y_train : training set (preprocessed beats)
    X_test, y_test   : test set (preprocessed beats)
    get_embeddings   : function mapping np.ndarray [B, L] -> np.ndarray [B, D]
                       using the encoder of interest.
    idx2cls          : mapping from class index to label string.
    output_dir       : base output directory (HuggingFace run dir).
    prefix           : short string to prefix saved artifacts, e.g. "mae".
    """

    print(f"\n=== Zero-shot KNN + t-SNE evaluation ({prefix}) ===")

    # ---- Raw-space KNN (train -> test) ----
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    knn_raw = KNeighborsClassifier(n_neighbors=5)
    knn_raw.fit(X_train_flat, y_train)
    raw_acc = (knn_raw.predict(X_test_flat) == y_test).mean()
    print(f"[{prefix}] KNN accuracy in raw space (test): {raw_acc:.4f}")

    # ---- Encoder-space embeddings (train & test) ----
    Z_train = get_embeddings(X_train)  # [N_train, D]
    Z_test = get_embeddings(X_test)    # [N_test, D]

    knn_latent = KNeighborsClassifier(n_neighbors=5)
    knn_latent.fit(Z_train, y_train)
    latent_acc = (knn_latent.predict(Z_test) == y_test).mean()
    print(f"[{prefix}] KNN accuracy in encoder space (test): {latent_acc:.4f}")

    # ---- t-SNE on test encoder embeddings ----
    tsne_dir = Path(output_dir)
    tsne_dir.mkdir(parents=True, exist_ok=True)

    max_points = 2000
    n_test = len(Z_test)
    if n_test > max_points:
        idx_vis = np.random.choice(n_test, size=max_points, replace=False)
        Z_vis = Z_test[idx_vis]
        y_vis = y_test[idx_vis]
    else:
        Z_vis = Z_test
        y_vis = y_test

    if len(Z_vis) < 2:
        print(f"[{prefix}] Not enough test points for t-SNE, skipping.")
        return

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="random")
    Z_tsne = tsne.fit_transform(Z_vis)

    plt.figure(figsize=(6, 6))
    classes = sorted(set(int(c) for c in np.unique(y_vis)))
    for c in classes:
        mask = y_vis == c
        if not np.any(mask):
            continue
        plt.scatter(Z_tsne[mask, 0], Z_tsne[mask, 1], s=5, label=idx2cls.get(c, str(c)))
    plt.legend()
    plt.title(f"{prefix.upper()} encoder t-SNE (test embeddings)")
    plt.tight_layout()
    tsne_path = tsne_dir / f"{prefix}_encoder_tsne.png"
    plt.savefig(tsne_path)
    plt.close()


__all__ = [
    "SEED",
    "ECGMAEDataset",
    "mae_collator",
    "cls_collator",
    "add_common_ecg_cli_args",
    "evaluate_knn_and_tsne_on_test",
    "make_training_args",
]
