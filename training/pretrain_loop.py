#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.metrics import classification_report, confusion_matrix
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


def run_finetune(
    backbone,
    esc50_data,
    training_args,
    transform,
    output_dir,
):
    finetune_args = copy.deepcopy(training_args)
    finetune_args.output_dir = str(output_dir)

    x_train = esc50_data["X_train"]
    x_valid = esc50_data["X_valid"]
    x_test = esc50_data["X_test"]
    y_train = esc50_data["y_train"]
    y_valid = esc50_data["y_valid"]
    y_test = esc50_data["y_test"]
    label_names = esc50_data["label_names"]
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

    trainer = Trainer(
        model=clf_model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=cls_collator,
        compute_metrics=compute_metrics,
    )

    if finetune_args.num_train_epochs > 0:
        trainer.train()

    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)

    pred_output = trainer.predict(test_dataset)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids
    label_ids = list(range(n_classes))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)

    metrics_path = Path(output_dir) / "finetune_metrics.txt"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        "\n".join(
            [
                "Validation metrics:",
                str(val_metrics),
                "",
                "Test metrics:",
                str(test_metrics),
                "",
                "Classification report:",
                report,
                "",
                "Confusion matrix:",
                np.array2string(matrix),
            ]
        )
        + "\n"
    )

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
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

    if checkpoint is not None:
        if checkpoint.endswith(".safetensors"):
            state_dict = load_file(checkpoint)
        else:
            state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

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
    trainer.train()
    return trainer
