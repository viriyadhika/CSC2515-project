#!/usr/bin/env python3
"""
MAE-style pretraining for AST-shaped audio spectrogram transformers.

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
from models.mae_model import AudioASTMAE
from novel.mae_lib import mae_collator
from training.pretrain_loop import run_finetune, run_pretrain_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audioset_dir", type=str, default="data/audioset_5000")
    parser.add_argument("--model_name", type=str, default=AST_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="data/audio_ast_mae_runs")
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
    parser.add_argument("--fshape", type=int, default=16)
    parser.add_argument("--tshape", type=int, default=16)
    parser.add_argument("--mask_patch", type=int, default=100)
    parser.add_argument("--decoder_dim", type=int, default=256)
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

    if not datasets:
        raise RuntimeError("No audio sources available for pretraining.")

    full_dataset = ConcatDataset(datasets)
    valid_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - valid_size
    pretrain_train_dataset, pretrain_valid_dataset = random_split(
        full_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(SEED),
    )

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
    mae_model = AudioASTMAE(
        config=config,
        num_mel_bins=args.num_mel_bins,
        target_length=args.target_length,
        fshape=args.fshape,
        tshape=args.tshape,
        mask_patch=args.mask_patch,
        decoder_dim=args.decoder_dim,
    )

    idx2cls = {i: label for i, label in enumerate(finetune_data["label_names"])}

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
    pretrain_args.metric_for_best_model = "eval_loss"
    pretrain_args.greater_is_better = False
    pretrain_args.weight_decay = args.weight_decay

    run_pretrain_loop(
        model=mae_model,
        train_dataset=pretrain_train_dataset,
        valid_dataset=pretrain_valid_dataset,
        collator=mae_collator,
        training_args=pretrain_args,
        eval_callback=evaluate_epoch,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
