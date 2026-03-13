#!/usr/bin/env python3
"""
MAE-style ECG pretraining script that operates on STFT spectrograms.

This mirrors the pipeline in `novel/mae.py` but replaces the 1D Conv tokenizer
with a 2D Conv over [freq, time] patches of the STFT magnitude:

    spec = torch.stft(
        x,               # [B, L]
        n_fft=64,
        hop_length=16,
        return_complex=True,
    )
    spec_mag = spec.abs()        # [B, F, T]

The model then uses:

    self.patch_embed = nn.Conv2d(
        in_channels=1,
        out_channels=embed_dim,
        kernel_size=(freq_patch, time_patch),
        stride=(freq_patch, time_patch),
    )

and reshapes the Conv2d output into [B, N, embed_dim] tokens for a Transformer
encoder/decoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import Trainer
from dataclasses import dataclass

from common.lib import (
    SEED,
    IDX2CLS,
    seed_everything,
    extract_beats_and_rr,
    low_pass_filter,
    maybe_augment_noise,
    ECGRRDataset,
    make_training_args,
    compute_metrics,
    percent_trained,
    preprocess_beats_and_balance,
)
from novel.mae_lib import (
    ECGMAEDataset,
    mae_collator,
    cls_collator,
    add_common_ecg_cli_args,
)


@dataclass
class SpectrogramLayout:
    """Simple container describing the 2D patch layout for a spectrogram."""

    freq_bins: int
    time_frames: int
    freq_patch: int
    time_patch: int

    @property
    def num_freq_patches(self) -> int:
        return self.freq_bins // self.freq_patch

    @property
    def num_time_patches(self) -> int:
        return self.time_frames // self.time_patch

    @property
    def num_patches(self) -> int:
        return self.num_freq_patches * self.num_time_patches

    @property
    def patch_area(self) -> int:
        return self.freq_patch * self.time_patch


def infer_spectrogram_layout(
    seq_len: int,
    n_fft: int,
    hop_length: int,
    freq_patch: int,
    time_patch: int,
    center: bool = True,
    device: torch.device | None = None,
) -> SpectrogramLayout:
    """
    Run a dummy STFT to figure out the [F, T] grid, then compute the
    effective cropped dimensions that are exactly tiled by non-overlapping
    (freq_patch, time_patch) patches.
    """

    if device is None:
        device = torch.device("cpu")

    with torch.no_grad():
        dummy = torch.zeros(1, seq_len, device=device)
        spec = torch.stft(
            dummy,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            center=center,
        )  # [1, F, T]

    _, F, T = spec.shape

    freq_bins = (F // freq_patch) * freq_patch
    time_frames = (T // time_patch) * time_patch

    return SpectrogramLayout(
        freq_bins=freq_bins,
        time_frames=time_frames,
        freq_patch=freq_patch,
        time_patch=time_patch,
    )


def compute_spec_mag(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    center: bool = True,
) -> torch.Tensor:
    """
    Compute STFT magnitude for a batch of 1D signals.

    Parameters
    ----------
    x : Tensor
        Shape [B, 1, L] or [B, L].
    """

    if x.dim() == 3:
        x = x.squeeze(1)  # [B, L]

    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        center=center,
    )  # [B, F, T]
    return spec.abs()


def spec_patchify(
    spec_mag: torch.Tensor,
    layout: SpectrogramLayout,
) -> torch.Tensor:
    """
    Turn a spectrogram magnitude into flattened 2D patches.

    spec_mag : [B, F, T]
    returns  : [B, N, P] where N = num_patches, P = freq_patch * time_patch
    """

    B, F, T = spec_mag.shape

    F_eff = layout.freq_bins
    T_eff = layout.time_frames

    x = spec_mag[:, :F_eff, :T_eff]  # [B, F_eff, T_eff]

    # [B, Fp, fp, Tp, tp] then -> [B, Fp, Tp, fp, tp]
    Fp = layout.num_freq_patches
    Tp = layout.num_time_patches

    x = x.reshape(B, Fp, layout.freq_patch, Tp, layout.time_patch)
    x = x.permute(0, 1, 3, 2, 4)  # [B, Fp, Tp, fp, tp]

    patches = x.reshape(B, Fp * Tp, layout.patch_area)
    return patches


def mae_random_masking(
    x: torch.Tensor,
    mask_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard MAE-style random masking over the patch dimension.

    x : [B, N, D]
    returns:
        x_masked : [B, N_keep, D]
        mask     : [B, N] (1 for masked, 0 for visible)
        ids_restore : [B, N] indices to restore original ordering
    """

    B, N, D = x.shape
    len_keep = int(N * (1.0 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]

    x_masked = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
    )

    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def mae_forward_loss(
    loss_fn: nn.Module,
    target: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Patch-wise reconstruction loss averaged over masked patches only.

    target : [B, N, P]
    pred   : [B, N, P]
    mask   : [B, N] (1 for masked patches)
    """

    loss = loss_fn(pred, target).mean(dim=-1)  # [B, N]
    loss = (loss * mask).sum() / mask.sum()
    return loss




class ECGMAEFreq(nn.Module):
    """
    MAE variant that reconstructs STFT magnitude patches instead of raw samples.
    """

    def __init__(
        self,
        seq_len: int = 198,
        n_fft: int = 64,
        hop_length: int = 16,
        freq_patch: int = 4,
        time_patch: int = 3,
        embed_dim: int = 64,
        decoder_dim: int = 32,
        nhead: int = 8,
        mask_ratio: float = 0.6,
        n_layer: int = 4,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.n_layer = n_layer

        # Figure out STFT [F, T] grid and patch tiling for this seq_len.
        layout: SpectrogramLayout = infer_spectrogram_layout(
            seq_len=seq_len,
            n_fft=n_fft,
            hop_length=hop_length,
            freq_patch=freq_patch,
            time_patch=time_patch,
        )
        self.layout = layout

        # Conv2d tokenizer over [freq, time] grid
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(layout.freq_patch, layout.time_patch),
            stride=(layout.freq_patch, layout.time_patch),
        )

        self.num_patches = layout.num_patches
        self.patch_size = layout.patch_area  # for compatibility with callbacks

        # Positional embeddings for encoder tokens
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layer)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim)
        )

        dec = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=4,
            dim_feedforward=decoder_dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(dec, num_layers=1)
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Predict flattened spectrogram patch values
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_size)
        self.loss_fn = nn.MSELoss(reduction="none")

    def _spec_mag(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, L]
        returns spec_mag : [B, F, T]
        """
        spec_mag = compute_spec_mag(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
        )
        # Crop to the effective grid we precomputed
        spec_mag = spec_mag[
            :,
            : self.layout.freq_bins,
            : self.layout.time_frames,
        ]
        return spec_mag

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, L] (time-domain)
        returns patches of STFT magnitude: [B, N, P]
        """
        spec_mag = self._spec_mag(x)  # [B, F, T]
        patches = spec_patchify(spec_mag, self.layout)  # [B, N, P]
        return patches

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, L] (time-domain)
        returns tokens : [B, N, D]
        """
        spec_mag = self._spec_mag(x)  # [B, F, T]
        spec_mag = spec_mag.unsqueeze(1)  # [B, 1, F, T]

        tokens_2d = self.patch_embed(spec_mag)  # [B, D, Fp, Tp]
        tokens = tokens_2d.flatten(2).transpose(1, 2)  # [B, N, D]
        return tokens

    def forward_encoder(self, x: torch.Tensor):
        """
        x : [B, 1, L]
        """
        target = self.patchify(x).detach()  # [B, N, P]
        tokens = self.tokenize(x)  # [B, N, D]

        tokens = tokens + self.pos
        x_masked, mask, ids_restore = mae_random_masking(tokens, self.mask_ratio)

        latent = self.encoder(x_masked)
        latent = self.encoder_norm(latent)
        return latent, target, mask, ids_restore

    def forward_decoder(self, latent, ids_restore):
        x = self.decoder_embed(latent)  # [B, N_keep, D_dec]
        B, N_keep, D = x.shape
        N = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(B, N - N_keep, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # [B, N, D]

        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
        )

        x_ = x_ + self.decoder_pos
        x_ = self.decoder(x_)
        x_ = self.decoder_norm(x_)

        pred = self.decoder_pred(x_)  # [B, N, P]
        return pred

    def forward_loss(self, target, pred, mask):
        return mae_forward_loss(self.loss_fn, target, pred, mask)

    def forward(self, x=None, labels=None):
        latent, target, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(target, pred, mask)
        return {"loss": loss, "logits": pred}

    def build_classifier(self, n_classes: int = 5, class_weights: torch.Tensor | None = None):
        """
        Build a spectrogram classifier, initialized from the MAE encoder,
        mirroring `TinyTransformerMAEFinetune`.
        """
        model = TinyTransformerMAEFinetuneFreq(
            n_layer=self.n_layer,
            embed_dim=self.embed_dim,
            n_classes=n_classes,
            layout=self.layout,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            class_weights=class_weights,
        )

        model.patch_embed.load_state_dict(self.patch_embed.state_dict())
        model.pos.data.copy_(self.pos.data)
        model.encoder.load_state_dict(self.encoder.state_dict())
        model.final_norm.load_state_dict(self.encoder_norm.state_dict())

        return model


class TinyTransformerMAEFinetuneFreq(nn.Module):
    """
    Spectrogram classifier initialized from `ECGMAEFreq` encoder.
    """

    def __init__(
        self,
        n_layer: int = 1,
        embed_dim: int = 64,
        n_classes: int = 5,
        layout: SpectrogramLayout | None = None,
        n_fft: int = 64,
        hop_length: int = 16,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        assert layout is not None, "layout must be provided"
        self.layout = layout
        self.n_fft = n_fft
        self.hop_length = hop_length

        num_patches = layout.num_patches

        # SAME tokenizer shape as MAE (Conv2d)
        self.patch_embed = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(layout.freq_patch, layout.time_patch),
            stride=(layout.freq_patch, layout.time_patch),
        )

        self.pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.2,
        )

        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layer)
        self.final_norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, n_classes)

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def _spec_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, L]
        returns tokens : [B, N, D]
        """
        spec_mag = compute_spec_mag(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
        )  # [B, F, T]
        spec_mag = spec_mag[
            :,
            : self.layout.freq_bins,
            : self.layout.time_frames,
        ]
        spec_mag = spec_mag.unsqueeze(1)  # [B, 1, F, T]

        tokens_2d = self.patch_embed(spec_mag)  # [B, D, Fp, Tp]
        tokens = tokens_2d.flatten(2).transpose(1, 2)  # [B, N, D]
        return tokens

    def forward(self, x=None, labels=None):
        # x: [B, 1, L]
        x = self._spec_tokens(x)
        x = x + self.pos[:, : x.size(1), :]

        x = self.encoder(x)
        x = self.final_norm(x)

        x = x.mean(dim=1)

        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def main():
    parser = argparse.ArgumentParser()
    add_common_ecg_cli_args(parser, output_dir_default="./tiny_ecg_mae_freq_runs")

    parser.add_argument("--mask_ratio", type=float, default=0.6)

    # STFT + patching hyperparameters
    parser.add_argument("--seq_len", type=int, default=198)
    parser.add_argument("--n_fft", type=int, default=64)
    parser.add_argument("--hop_length", type=int, default=16)
    parser.add_argument("--freq_patch", type=int, default=4)
    parser.add_argument("--time_patch", type=int, default=3)

    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--decoder_dim", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=4)

    args = parser.parse_args()

    seed_everything(SEED)

    X, RR, y = extract_beats_and_rr(args.folder, pre_process=low_pass_filter)
    X, y = preprocess_beats_and_balance(
        X,
        y,
        per_beat_fn=None,
        target_size=None,
        seed=SEED,
        n_classes=5,
    )

    print(f"Loaded beats: {len(y)}")
    class_counts = {IDX2CLS[i]: int((y == i).sum()) for i in range(5)}
    print("Class counts:", class_counts)

    # Same 7:1:2 split as TinyTransformer / original MAE script
    X_train, X_tmp, RR_train, RR_tmp, y_train, y_tmp = train_test_split(
        X, RR, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_valid, X_test, RR_valid, RR_test, y_valid, y_test = train_test_split(
        X_tmp, RR_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=SEED
    )

    if args.use_noise_aug:
        X_train = maybe_augment_noise(X_train, args.nstdb_folder, args.snr_db)

    mae_train_dataset = ECGMAEDataset(X_train)
    mae_valid_dataset = ECGMAEDataset(X_valid)

    print("\n=== Stage 1: Spectrogram MAE pretraining ===")
    mae_model = ECGMAEFreq(
        seq_len=args.seq_len,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        freq_patch=args.freq_patch,
        time_patch=args.time_patch,
        embed_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        nhead=args.nhead,
        mask_ratio=args.mask_ratio,
        n_layer=args.n_layer,
    )

    mae_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "mae_pretrain"),
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        lr=args.pretrain_lr,
        seed=SEED,
    )
    mae_args.metric_for_best_model = "eval_loss"
    mae_args.greater_is_better = False

    if args.checkpoint is None:
        mae_trainer = Trainer(
            model=mae_model,
            args=mae_args,
            train_dataset=mae_train_dataset,
            eval_dataset=mae_valid_dataset,
            data_collator=mae_collator,
        )
        mae_trainer.train()

        print("Spectrogram MAE validation:")
        print(mae_trainer.evaluate())
    else:
        from safetensors.torch import load_file

        state_dict = load_file(args.checkpoint)
        mae_model.load_state_dict(state_dict)

    # ===== Stage 2: classifier finetuning (spectrogram TinyTransformer) =====
    print("\n=== Stage 2: spectrogram classifier finetuning ===")

    X_train_ft, y_train_ft, RR_train_ft = percent_trained(
        X_train, y_train, RR_train, args
    )

    class_counts_ft = np.bincount(y_train_ft, minlength=5).astype(np.float32)
    class_weights_ft_np = class_counts_ft.sum() / (
        len(class_counts_ft) * class_counts_ft + 1e-8
    )
    class_weights_ft = torch.tensor(class_weights_ft_np, dtype=torch.float32)

    clf_model = mae_model.build_classifier(
        n_classes=len(class_counts_ft),
        class_weights=class_weights_ft if args.balanced_weight else None,
    )

    train_dataset = ECGRRDataset(X_train_ft, RR_train_ft, y_train_ft)
    valid_dataset = ECGRRDataset(X_valid, RR_valid, y_valid)
    test_dataset = ECGRRDataset(X_test, RR_test, y_test)

    finetune_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "finetune"),
        epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        lr=args.finetune_lr,
        seed=SEED,
    )
    finetune_args.metric_for_best_model = "macro_f1"
    finetune_args.greater_is_better = True

    clf_trainer = Trainer(
        model=clf_model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=cls_collator,
        compute_metrics=compute_metrics,
    )
    clf_trainer.train()

    print("Validation metrics:")
    val_metrics = clf_trainer.evaluate()
    print(val_metrics)

    print("Test metrics:")
    test_metrics = clf_trainer.evaluate(eval_dataset=test_dataset)
    print(test_metrics)

    pred_output = clf_trainer.predict(test_dataset)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=[IDX2CLS[i] for i in range(5)],
            zero_division=0,
        )
    )
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
        )
    )


if __name__ == "__main__":
    main()

