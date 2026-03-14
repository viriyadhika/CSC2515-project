#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback

from common.lib import compute_metrics
from novel.mae_lib import cls_collator


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden_dim: int = 512,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOTeacherUpdateCallback(TrainerCallback):
    def __init__(self, momentum: float = 0.996):
        self.momentum = momentum

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            model.update_teacher(momentum=self.momentum)


def compute_feature_std(backbone: nn.Module, dataset, n_samples: int) -> float:
    backbone = backbone.eval()
    device = next(backbone.parameters()).device
    embeddings = []
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]["x"]
            x = sample.to(device, dtype=torch.float32).unsqueeze(0)
            z = backbone(x)
            embeddings.append(z)
    embeddings = torch.cat(embeddings, dim=0)
    return float(embeddings.std(dim=0).mean().item())


class DINOFeatureStdCallback(TrainerCallback):
    """Every `interval` epochs, compute mean feature std over first n_samples train embeddings."""

    def __init__(self, dataset, n_samples: int = 2000, interval: int = 10):
        self.dataset = dataset
        self.n_samples = min(n_samples, len(dataset))
        self.interval = interval

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or int(state.epoch) % self.interval != 0:
            return
        mean_std = compute_feature_std(model.student_backbone, self.dataset, self.n_samples)
        print(f"Epoch {int(state.epoch)} — Mean feature std: {mean_std:.4f}")


def dino_collator(batch):
    x1 = torch.stack([b["x1"] for b in batch])
    x2 = torch.stack([b["x2"] for b in batch])
    return {"x1": x1, "x2": x2}


def dino_pretrain_from_datasets(
    train_dataset,
    valid_dataset,
    dino_model: nn.Module,
    training_args,
    teacher_momentum: float = 0.996,
    feature_std_dataset=None,
):
    callbacks = [DINOTeacherUpdateCallback(momentum=teacher_momentum)]
    if feature_std_dataset is not None:
        callbacks.append(DINOFeatureStdCallback(feature_std_dataset))
    trainer = Trainer(
        model=dino_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=dino_collator,
        callbacks=callbacks,
    )
    trainer.train()
    return trainer


def dino_finetune_from_datasets(
    clf_model: nn.Module,
    train_dataset,
    valid_dataset,
    test_dataset,
    training_args,
    data_collator=cls_collator,
    compute_metrics_fn=compute_metrics,
):
    trainer = Trainer(
        model=clf_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    trainer.train()
    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    return trainer, val_metrics, test_metrics


__all__ = [
    "DINOHead",
    "DINOTeacherUpdateCallback",
    "DINOFeatureStdCallback",
    "compute_feature_std",
    "dino_collator",
    "dino_pretrain_from_datasets",
    "dino_finetune_from_datasets",
]
