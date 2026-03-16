#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import Trainer
from sklearn.metrics import classification_report, confusion_matrix

from common.dataloader import ESC50_AUDIO_RATE
from common.lib import compute_metrics
from models.audio_common import (
    WaveformArrayClassificationDataset,
    build_audio_classifier_from_backbone,
)
from novel.mae_lib import cls_collator


EVAL_EPOCHS = [15, 30]


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def run_ssl_epoch(
    model,
    dataloader,
    optimizer,
    device,
    teacher_momentum: float | None = None,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    running_loss = 0.0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            outputs = model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if teacher_momentum is not None and hasattr(model, "update_teacher"):
            model.update_teacher(momentum=teacher_momentum)

        running_loss += float(loss.item())

    return running_loss / max(1, len(dataloader))


def evaluate_ssl_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            running_loss += float(loss.item())
    return running_loss / max(1, len(dataloader))


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
    output_dir,
    epochs,
    batch_size,
    lr,
    weight_decay,
    eval_callback,
    checkpoint: str | None = None,
    teacher_momentum: float | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if checkpoint is not None:
        if checkpoint.endswith(".safetensors"):
            state_dict = load_file(checkpoint)
        else:
            state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_valid = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_ssl_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            teacher_momentum=teacher_momentum,
        )
        valid_loss = evaluate_ssl_epoch(model, valid_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")

        torch.save(model.state_dict(), output_path / "latest.pt")
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(model.state_dict(), output_path / "best.pt")

        if epoch in EVAL_EPOCHS:
            eval_callback(model, epoch)

    return history
