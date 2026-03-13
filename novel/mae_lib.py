#!/usr/bin/env python3
from __future__ import annotations

"""
MAE-specific datasets and collators used by the novel models.
"""

from typing import List, Dict
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset

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
        "--folder", type=str, default="mit-bih-arrhythmia-database-1.0.0"
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


__all__ = [
    "SEED",
    "ECGMAEDataset",
    "mae_collator",
    "cls_collator",
    "add_common_ecg_cli_args",
    "make_training_args",
]

