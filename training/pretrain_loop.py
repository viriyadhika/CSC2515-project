#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainerCallback

from common.dataloader import ESC50_AUDIO_RATE
from common.lib import compute_metrics
from models.audio_common import (
    WaveformArrayClassificationDataset,
    build_audio_classifier_from_backbone,
)
from novel.dino_utils import compute_feature_std
from novel.mae_lib import cls_collator


EVAL_EPOCHS = [15, 30]


class MilestoneEvalCallback(TrainerCallback):
    def __init__(self, eval_callback):
        self.eval_callback = eval_callback
        self.completed_epochs = set()

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.epoch is None:
            return
        epoch = int(round(state.epoch))
        if epoch in EVAL_EPOCHS and epoch not in self.completed_epochs:
            self.completed_epochs.add(epoch)
            self.eval_callback(model, epoch)


class CollapseLoggingCallback(TrainerCallback):
    def __init__(
        self,
        collapse_dataset,
        collapse_backbone_getter,
        output_dir,
        n_samples: int = 2000,
    ):
        self.collapse_dataset = collapse_dataset
        self.collapse_backbone_getter = collapse_backbone_getter
        self.n_samples = n_samples
        self.output_path = Path(output_dir) / "collapse_metrics.txt"

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.epoch is None:
            return
        epoch = int(round(state.epoch))
        mean_std = compute_feature_std(
            self.collapse_backbone_getter(model),
            self.collapse_dataset,
            n_samples=min(self.n_samples, len(self.collapse_dataset)),
        )
        print(f"Epoch {epoch}: mean_feature_std={mean_std:.6f}")
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(f"epoch={epoch},mean_feature_std={mean_std:.6f}\n")


class _DifferentialLRTrainer(Trainer):
    """Trainer with separate learning rates for backbone and head."""

    def __init__(
        self,
        *args,
        backbone_lr: float,
        head_lr: float,
        backbone_attr: str = "backbone",
        head_attr: str = "head",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._backbone_lr = backbone_lr
        self._head_lr = head_lr
        self._backbone_attr = backbone_attr
        self._head_attr = head_attr

    def create_optimizer(self):
        backbone_params = list(getattr(self.model, self._backbone_attr).parameters())
        head_params = list(getattr(self.model, self._head_attr).parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self._backbone_lr},
                {"params": head_params, "lr": self._head_lr},
            ],
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer


class FreezeBackboneCallback(TrainerCallback):
    """Freezes the backbone for the first `freeze_epochs` epochs, then releases it."""

    def __init__(self, freeze_epochs: int, backbone_attr: str = "backbone"):
        self.freeze_epochs = freeze_epochs
        self.backbone_attr = backbone_attr

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.freeze_epochs > 0 and model is not None:
            for p in getattr(model, self.backbone_attr).parameters():
                p.requires_grad_(False)
            print(f"{self.backbone_attr} frozen for first {self.freeze_epochs} epochs.")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        if epoch == self.freeze_epochs and model is not None:
            for p in getattr(model, self.backbone_attr).parameters():
                p.requires_grad_(True)
            print(f"{self.backbone_attr} unfrozen at epoch {epoch}.")


def run_finetune(
    backbone,
    esc50_data,
    training_args,
    transform,
    output_dir,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
    freeze_backbone_epochs: int = 0,
):
    finetune_args = copy.deepcopy(training_args)
    finetune_args.output_dir = str(output_dir)

    x_train = esc50_data["X_train"]
    x_valid = esc50_data["X_valid"]
    x_test = esc50_data["X_test"]
    y_train = esc50_data["y_train"]
    y_valid = esc50_data["y_valid"]
    y_test = esc50_data["y_test"]
    n_classes = int(esc50_data["n_classes"])

    train_dataset = WaveformArrayClassificationDataset(
        x_train,
        y_train,
        source_rate=ESC50_AUDIO_RATE,
        transform=transform,
    )
    valid_dataset = WaveformArrayClassificationDataset(
        x_valid,
        y_valid,
        source_rate=ESC50_AUDIO_RATE,
        transform=transform,
    )
    test_dataset = WaveformArrayClassificationDataset(
        x_test,
        y_test,
        source_rate=ESC50_AUDIO_RATE,
        transform=transform,
    )

    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights_np = class_counts.sum() / (len(class_counts) * class_counts + 1e-8)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    clf_model = build_audio_classifier_from_backbone(
        backbone=backbone,
        n_classes=n_classes,
        class_weights=class_weights,
    )

    callbacks = []
    if freeze_backbone_epochs > 0:
        callbacks.append(FreezeBackboneCallback(freeze_backbone_epochs))

    trainer_cls = Trainer
    trainer_kwargs = {}
    if backbone_lr is not None and head_lr is not None:
        trainer_cls = _DifferentialLRTrainer
        trainer_kwargs = {"backbone_lr": backbone_lr, "head_lr": head_lr,
                          "backbone_attr": "backbone", "head_attr": "head"}

    trainer = trainer_cls(
        model=clf_model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=cls_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks or None,
        **trainer_kwargs,
    )

    if finetune_args.num_train_epochs > 0:
        trainer.train()

    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run_pretrain_loop(
    model,
    train_dataset,
    valid_dataset,
    collator,
    training_args,
    eval_callback,
    collapse_dataset=None,
    collapse_backbone_getter=None,
    checkpoint: str | None = None,
    collapse_n_samples: int = 2000,
    extra_callbacks=None,
):
    output_path = Path(training_args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    callbacks = [MilestoneEvalCallback(eval_callback=eval_callback)]
    if collapse_dataset is not None and collapse_backbone_getter is not None:
        callbacks.append(
            CollapseLoggingCallback(
                collapse_dataset=collapse_dataset,
                collapse_backbone_getter=collapse_backbone_getter,
                output_dir=output_path,
                n_samples=collapse_n_samples,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    return trainer
