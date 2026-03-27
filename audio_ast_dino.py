#!/usr/bin/env python3
"""
DINO-style self-supervised learning for AST-shaped audio transformers.

Pre-trains for --pretrain_epochs epochs, evaluating KNN at epochs 15 and 30,
then runs a single --finetune_epochs supervised finetune from the pre-trained
backbone. All metrics are written to {output_dir}/metrics.json and per-epoch
finetune logs are available in {output_dir}/final_finetune/trainer_state.json.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

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
from common.metrics_logger import MetricsLogger
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
    from pathlib import Path
    path = Path(checkpoint_path)
    if path.is_dir():
        if (path / "model.safetensors").exists():
            checkpoint_path = str(path / "model.safetensors")
        elif (path / "pytorch_model.bin").exists():
            checkpoint_path = str(path / "pytorch_model.bin")
        else:
            raise FileNotFoundError(f"No model file found in {path}")
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)


class AudioDINOPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, asymmetric: bool = False):
        self.base_dataset = base_dataset
        self.asymmetric = asymmetric

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

    def augment_strong(self, x: torch.Tensor) -> torch.Tensor:
        """Heavy time masking for student view (DINOSR-inspired).
        Masks ~50% of time frames in 3 contiguous blocks."""
        x = x.clone()
        total_time = x.shape[0]
        mask_width = total_time // 6
        for _ in range(3):
            start = int(torch.randint(0, max(1, total_time - mask_width + 1), (1,)).item())
            x[start : start + mask_width] = 0.0
        return x

    def __getitem__(self, idx: int):
        x = self.base_dataset[idx]["x"]
        if self.asymmetric:
            # Student sees heavily masked view, teacher sees clean original
            return {"x1": self.augment_strong(x), "x2": x.clone()}
        return {"x1": self.augment(x), "x2": self.augment(x)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audioset_dir", type=str, default="data/audioset_20000")
    parser.add_argument("--model_name", type=str, default=AST_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default="data/runs/dino")
    parser.add_argument("--teacher_momentum", type=float, default=0.99)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients over N steps; effective batch = batch_size * N")
    parser.add_argument("--teacher_temp", type=float, default=0.04)
    parser.add_argument("--student_temp", type=float, default=0.1)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--asymmetric_aug", action="store_true",
                        help="Student sees heavily masked spectrogram, teacher sees clean (DINOSR-inspired)")
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--finetune_epochs", type=int, default=45)
    parser.add_argument("--pretrain_batch_size", type=int, default=8)
    parser.add_argument("--finetune_batch_size", type=int, default=8)
    parser.add_argument("--pretrain_lr", type=float, default=1e-4)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_mel_bins", type=int, default=128)
    parser.add_argument("--target_length", type=int, default=1024)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--dataset_mean", type=float, default=None, help="Override computed mean")
    parser.add_argument("--dataset_std", type=float, default=None, help="Override computed std")
    parser.add_argument("--percent_train", type=float, default=100.0)
    parser.add_argument("--balance_target_size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    seed_everything(SEED)
    metrics_logger = MetricsLogger(args.output_dir)

    # 1. Discover data sources before building the transform so we can fit stats.
    audioset_files = find_audio_files(args.audioset_dir)
    if audioset_files:
        print(f"Found {len(audioset_files)} audio files in {args.audioset_dir}")
    else:
        print(f"No audio files found in {args.audioset_dir}, skipping.")

    esc50_data = AudioLoader(args).load()
    esc50_waveforms = esc50_data["X_train"]
    print(f"Loaded {len(esc50_waveforms)} ESC-50 train clips for pre-training.")

    # 2. Build transforms: one fitted on the full pretraining corpus (ESC-50 +
    #    AudioSet) for pretraining, and one fitted on ESC-50 only for finetuning.
    #    This mirrors the SSAST paper, which uses separate normalization stats for
    #    the pretraining dataset and each downstream dataset.
    def _make_transform(mean=None, std=None):
        return LogMelPadCrop(
            sample_rate=args.sample_rate,
            n_mels=args.num_mel_bins,
            target_length=args.target_length,
            dataset_mean=mean,
            dataset_std=std,
        )

    if args.dataset_mean is not None and args.dataset_std is not None:
        pretrain_transform = _make_transform(args.dataset_mean, args.dataset_std)
        finetune_transform = _make_transform(args.dataset_mean, args.dataset_std)
    else:
        pretrain_transform = _make_transform()
        pretrain_transform.fit(
            waveforms=esc50_waveforms,
            source_rate=ESC50_AUDIO_RATE,
            audio_files=audioset_files or None,
        )
        finetune_transform = _make_transform()
        finetune_transform.fit(waveforms=esc50_waveforms, source_rate=ESC50_AUDIO_RATE)

    # 3. Build datasets with the fitted transforms.
    datasets = []
    if audioset_files:
        datasets.append(FolderAudioPretrainDataset(audioset_files, pretrain_transform))
    datasets.append(
        WaveformArrayPretrainDataset(
            esc50_waveforms,
            source_rate=ESC50_AUDIO_RATE,
            transform=pretrain_transform,
        )
    )

    full_dataset = ConcatDataset(datasets)
    valid_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - valid_size
    base_train_dataset, base_valid_dataset = random_split(
        full_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    dino_train_dataset = AudioDINOPretrainDataset(base_train_dataset, asymmetric=args.asymmetric_aug)
    dino_valid_dataset = AudioDINOPretrainDataset(base_valid_dataset, asymmetric=args.asymmetric_aug)

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
        transform=finetune_transform,
    )
    embedding_test_dataset = WaveformArrayClassificationDataset(
        finetune_data["X_test"],
        finetune_data["y_test"],
        source_rate=ESC50_AUDIO_RATE,
        transform=finetune_transform,
    )
    idx2cls = {i: label for i, label in enumerate(finetune_data["label_names"])}

    config = AutoConfig.from_pretrained(args.model_name)
    dino_model = AudioASTDINO(
        config=config,
        out_dim=args.out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
    )
    if args.checkpoint is not None:
        load_checkpoint_into_model(dino_model, args.checkpoint)
        print(f"Loaded model from checkpoint: {args.checkpoint}")

    # Initial KNN before any training
    initial_snapshot = evaluate_embedding_snapshots(
        backbone=dino_model.clone_backbone(),
        train_dataset=embedding_train_dataset,
        y_train=finetune_data["y_train"],
        test_dataset=embedding_test_dataset,
        y_test=finetune_data["y_test"],
        output_dir=Path(args.output_dir) / "epoch_0",
        idx2cls=idx2cls,
        batch_size=max(args.pretrain_batch_size, args.finetune_batch_size),
        skip_viz=True,
    )
    metrics_logger.set("initial_knn_acc", initial_snapshot["knn_accuracy"])
    print(f"Initial KNN accuracy: {initial_snapshot['knn_accuracy']:.4f}")

    def evaluate_epoch(model, epoch: int) -> None:
        """KNN-only milestone evaluation during pre-training."""
        snapshot = evaluate_embedding_snapshots(
            backbone=model.clone_backbone(),
            train_dataset=embedding_train_dataset,
            y_train=finetune_data["y_train"],
            test_dataset=embedding_test_dataset,
            y_test=finetune_data["y_test"],
            output_dir=Path(args.output_dir) / f"epoch_{epoch}",
            idx2cls=idx2cls,
            batch_size=max(args.pretrain_batch_size, args.finetune_batch_size),
            skip_viz=True,
        )
        metrics_logger.set_milestone(epoch=epoch, knn_acc=snapshot["knn_accuracy"])
        print(f"Epoch {epoch} KNN accuracy: {snapshot['knn_accuracy']:.4f}")

    # Pre-training
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
    pretrain_args.gradient_accumulation_steps = args.gradient_accumulation_steps

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

    # Final supervised fine-tune (single run, full budget)
    if args.finetune_epochs > 0:
        final_dir = Path(args.output_dir) / "final_finetune"
        final_args = make_training_args(
            output_dir=str(final_dir),
            epochs=args.finetune_epochs,
            batch_size=args.finetune_batch_size,
            lr=args.finetune_lr,
            seed=SEED,
        )
        final_args.metric_for_best_model = "eval_loss"
        final_args.greater_is_better = False
        final_args.max_grad_norm = 1.0

        freeze_backbone_epochs = max(1, int(0.2 * args.finetune_epochs))
        result = run_finetune(
            backbone=dino_model.clone_backbone(),
            esc50_data=finetune_data,
            training_args=final_args,
            transform=finetune_transform,
            output_dir=final_dir,
            backbone_lr=1e-5,
            head_lr=1e-4,
            freeze_backbone_epochs=freeze_backbone_epochs,
        )
        val_m = result["val_metrics"]
        test_m = result["test_metrics"]
        metrics_logger.set_final(
            val_acc=val_m.get("eval_accuracy"),
            val_f1=val_m.get("eval_macro_f1"),
            test_acc=test_m.get("eval_accuracy"),
            test_f1=test_m.get("eval_macro_f1"),
        )
        print(f"Final val accuracy:  {val_m.get('eval_accuracy')}")
        print(f"Final test accuracy: {test_m.get('eval_accuracy')}")


if __name__ == "__main__":
    main()
