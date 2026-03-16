#!/usr/bin/env python3
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from transformers import AutoModel

from mae_freq import SpectrogramLayout, mae_forward_loss
from models.audio_common import extract_last_hidden_state


def compatible_num_heads(embed_dim: int, preferred_heads: int) -> int:
    preferred_heads = max(1, int(preferred_heads))
    if embed_dim % preferred_heads == 0:
        return preferred_heads

    gcd_heads = math.gcd(embed_dim, preferred_heads)
    if gcd_heads > 0:
        return gcd_heads

    for heads in range(min(embed_dim, preferred_heads), 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


def random_mask_by_count(
    x: torch.Tensor,
    mask_patch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, n_tokens, dim = x.shape
    if mask_patch <= 0 or mask_patch >= n_tokens:
        raise ValueError(f"mask_patch must be in [1, {n_tokens - 1}], got {mask_patch}")

    len_keep = n_tokens - mask_patch
    noise = torch.rand(bsz, n_tokens, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]

    x_masked = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, dim),
    )

    mask = torch.ones(bsz, n_tokens, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class AudioASTMAE(nn.Module):
    def __init__(
        self,
        config,
        num_mel_bins: int,
        target_length: int,
        fshape: int,
        tshape: int,
        mask_patch: int,
        decoder_dim: int = 256,
    ):
        super().__init__()

        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.mask_patch = mask_patch
        self.base_config = config
        self.hidden_size = int(config.hidden_size)
        self.decoder_dim = decoder_dim

        self.layout = SpectrogramLayout(
            freq_bins=(num_mel_bins // fshape) * fshape,
            time_frames=(target_length // tshape) * tshape,
            freq_patch=fshape,
            time_patch=tshape,
        )
        self.num_patches = self.layout.num_patches
        self.patch_size = self.layout.patch_area

        self.backbone = AutoModel.from_config(config)

        self.decoder_embed = nn.Linear(self.hidden_size, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        decoder_heads = compatible_num_heads(
            decoder_dim,
            max(1, int(getattr(config, "num_attention_heads", 8)) // 2),
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_size)
        self.loss_fn = nn.MSELoss(reduction="none")

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, _ = x.shape
        x = x[:, : self.layout.time_frames, : self.layout.freq_bins]
        tp = self.layout.num_time_patches
        fp = self.layout.num_freq_patches
        x = x.reshape(bsz, tp, self.layout.time_patch, fp, self.layout.freq_patch)
        x = x.permute(0, 1, 3, 2, 4)
        return x.reshape(bsz, tp * fp, self.patch_size)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, : self.layout.time_frames, : self.layout.freq_bins]
        embeddings = self.backbone.embeddings(x)
        special_tokens = embeddings.shape[1] - self.num_patches
        if special_tokens < 0:
            raise ValueError(
                f"Backbone returned too few tokens: got {embeddings.shape[1]}, expected at least {self.num_patches}"
            )
        return embeddings[:, special_tokens:, :]

    def forward_encoder(self, x: torch.Tensor):
        target = self.patchify(x).detach()
        tokens = self.tokenize(x)
        x_masked, mask, ids_restore = random_mask_by_count(tokens, self.mask_patch)
        encoder_outputs = self.backbone.encoder(x_masked)
        latent = extract_last_hidden_state(encoder_outputs)
        if hasattr(self.backbone, "layernorm"):
            latent = self.backbone.layernorm(latent)
        return latent, target, mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(latent)
        bsz, n_keep, dim = x.shape
        n_tokens = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(bsz, n_tokens - n_keep, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, dim),
        )
        x_ = x_ + self.decoder_pos
        x_ = self.decoder(x_)
        x_ = self.decoder_norm(x_)
        return self.decoder_pred(x_)

    def forward(self, x=None, labels=None):
        latent, target, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        loss = mae_forward_loss(self.loss_fn, target, pred, mask)
        return {"loss": loss, "logits": pred}

    def clone_backbone(self) -> nn.Module:
        return copy.deepcopy(self.backbone)
