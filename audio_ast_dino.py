#!/usr/bin/env python3
"""
DINO-style self-supervised learning for AST-shaped audio transformers.

This script pretrains epoch-by-epoch and evaluates representation quality at
epochs 15 and 30 via:
- zero-shot KNN
- t-SNE / PCA grouped plots
- temporary ESC-50 finetuning from the current backbone
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, random_split
from transformers import AutoConfig

from common.dataloader import AudioLoader, ESC50_AUDIO_RATE
from common.lib import (
    SEED,
    balance_classes,
    make_training_args,
    percent_trained,
    seed_everything,
)
from evaluation.embedding_eval import evaluate_embedding_snapshots
from models.audio_common import (
    AST_MODEL_NAME,
    FolderAudioPretrainDataset,
    LogMelPadCrop,
    WaveformArrayClassificationDataset,
    WaveformArrayPretrainDataset,
    find_audio_files,
)
from models.dino_model import AudioASTDINO
from novel.dino_utils import DINOTeacherUpdateCallback, dino_collator
from training.pretrain_loop import run_finetune, run_pretrain_loop


def load_checkpoint_into_model(model, checkpoint_path: str) -> None:
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audioset_dir", type=str, default="data/audioset_5000")
    parser.add_argument("--model_name", type=str, default=AST_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="data/audio_ast_dino_runs")
    parser.add_argument("--teacher_momentum", type=float, default=0.996)
    parser.add_argument("--out_dim", type=int, default=256)
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
    parser.add_argument("--percent_train", type=float, default=100.0)
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
    esc50_waveforms = esc50_data["X_train"]
    datasets.append(
        WaveformArrayPretrainDataset(
            esc50_waveforms,
            source_rate=ESC50_AUDIO_RATE,
            transform=transform,
        )
    )
    print(f"Loaded {len(esc50_waveforms)} ESC-50 train clips from AudioLoader.")

    full_dataset = ConcatDataset(datasets)
    valid_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - valid_size
    base_train_dataset, base_valid_dataset = random_split(
        full_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    dino_train_dataset = AudioDINOPretrainDataset(base_train_dataset)
    dino_valid_dataset = AudioDINOPretrainDataset(base_valid_dataset)

    finetune_data = copy.deepcopy(esc50_data)
    x_train, y_train = percent_trained(finetune_data["X_train"], finetune_data["y_train"], args)
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
    dino_model = AudioASTDINO(config=config, out_dim=args.out_dim)
    if args.checkpoint is not None:
        load_checkpoint_into_model(dino_model, args.checkpoint)
        print(f"Loaded model from checkpoint: {args.checkpoint}")
    idx2cls = {i: label for i, label in enumerate(finetune_data["label_names"])}

    initial_dir = Path(args.output_dir) / "epoch_0"
    initial_snapshot = evaluate_embedding_snapshots(
        backbone=dino_model.clone_backbone(),
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
    pretrain_args.load_best_model_at_end = False
    pretrain_args.metric_for_best_model = None
    pretrain_args.weight_decay = args.weight_decay

    run_pretrain_loop(
        model=dino_model,
        train_dataset=dino_train_dataset,
        valid_dataset=dino_valid_dataset,
        collator=dino_collator,
        training_args=pretrain_args,
        eval_callback=evaluate_epoch,
        collapse_dataset=base_train_dataset,
        collapse_backbone_getter=lambda model: model.student_backbone,
        extra_callbacks=[DINOTeacherUpdateCallback(momentum=args.teacher_momentum)],
    )


if __name__ == "__main__":
    main()
