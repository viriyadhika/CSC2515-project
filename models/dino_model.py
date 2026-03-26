#!/usr/bin/env python3
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from models.audio_common import extract_last_hidden_state


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


class AudioASTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.hidden_size = int(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        hidden = extract_last_hidden_state(outputs)
        return hidden.mean(dim=1)


class AudioASTDINO(nn.Module):
    def __init__(self, config, out_dim: int = 256):
        super().__init__()
        self.base_config = config

        self.student_backbone = AudioASTBackbone(config)
        self.student_head = DINOHead(self.student_backbone.hidden_size, out_dim=out_dim)

        self.teacher_backbone = AudioASTBackbone(copy.deepcopy(config))
        self.teacher_head = DINOHead(self.teacher_backbone.hidden_size, out_dim=out_dim)

        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        for ps, pt in zip(
            self.student_backbone.parameters(), self.teacher_backbone.parameters()
        ):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor, momentum: float = 0.9) -> None:
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(momentum).add_(batch_center, alpha=1 - momentum)

    def dino_loss(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
    ) -> torch.Tensor:
        student_logp = F.log_softmax(student_out / student_temp, dim=-1)
        teacher_prob = F.softmax((teacher_out.detach() - self.center) / teacher_temp, dim=-1)
        return torch.sum(-teacher_prob * student_logp, dim=-1).mean()

    def forward(self, x1=None, x2=None, **kwargs):
        if x1 is None:
            x1 = kwargs.get("x1")
        if x2 is None:
            x2 = kwargs.get("x2")
        if x1 is None or x2 is None:
            raise ValueError("AudioASTDINO expects x1 and x2")

        student_feat1 = self.student_backbone(x1)
        student_out1 = self.student_head(student_feat1)
        student_feat2 = self.student_backbone(x2)
        student_out2 = self.student_head(student_feat2)

        with torch.no_grad():
            teacher_feat1 = self.teacher_backbone(x1)
            teacher_out1 = self.teacher_head(teacher_feat1)
            teacher_feat2 = self.teacher_backbone(x2)
            teacher_out2 = self.teacher_head(teacher_feat2)

        loss = 0.5 * (
            self.dino_loss(student_out1, teacher_out2)
            + self.dino_loss(student_out2, teacher_out1)
        )

        if self.training:
            with torch.no_grad():
                self.update_center(torch.cat([teacher_out1, teacher_out2], dim=0))

        return {"loss": loss, "logits": student_out1}

    def clone_backbone(self) -> nn.Module:
        return copy.deepcopy(self.student_backbone.model)
