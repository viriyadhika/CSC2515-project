#!/usr/bin/env python3
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
AST_MODEL_NAME = "Simon-Kotchou/ssast-small-patch-audioset-16-16"


def find_audio_files(root: str | None) -> list[str]:
    if root is None:
        return []

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    files = [
        str(path)
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    ]
    return sorted(set(files))


def extract_last_hidden_state(outputs):
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


class LogMelPadCrop:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        win_length: int = 400,
        hop_length: int = 160,
        target_length: int = 1024,
        top_db: float | None = 80.0,
        dataset_mean: float | None = None,
        dataset_std: float | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.top_db = top_db
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=top_db
        )

    def _to_raw_db(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Compute log-mel spectrogram without normalization."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        spec = self.melspec(waveform)
        spec = self.amplitude_to_db(spec)
        spec = spec.squeeze(0).transpose(0, 1)

        t, _ = spec.shape
        if t < self.target_length:
            spec = torch.nn.functional.pad(spec, (0, 0, 0, self.target_length - t))
        elif t > self.target_length:
            start = random.randint(0, t - self.target_length)
            spec = spec[start : start + self.target_length]

        return spec

    def fit(
        self,
        waveforms: np.ndarray | None = None,
        source_rate: int | None = None,
        audio_files: list[str] | None = None,
        n_samples: int = 500,
    ) -> "LogMelPadCrop":
        """Estimate dataset_mean and dataset_std from a sample of the data.

        Pass waveforms (numpy array) for in-memory sources and/or audio_files
        (list of paths) for on-disk sources. Samples up to n_samples from each.
        """
        specs = []

        if waveforms is not None and source_rate is not None:
            indices = random.sample(range(len(waveforms)), min(n_samples, len(waveforms)))
            for i in indices:
                w = torch.tensor(waveforms[i], dtype=torch.float32)
                specs.append(self._to_raw_db(w, source_rate))

        if audio_files:
            sampled = random.sample(audio_files, min(n_samples, len(audio_files)))
            for path in sampled:
                try:
                    w, sr = torchaudio.load(path)
                    specs.append(self._to_raw_db(w, sr))
                except Exception:
                    pass

        if not specs:
            raise RuntimeError(
                "LogMelPadCrop.fit() received no data. "
                "Pass waveforms= or audio_files=."
            )

        stacked = torch.stack(specs)
        self.dataset_mean = stacked.mean().item()
        self.dataset_std = stacked.std().item()
        print(
            f"LogMelPadCrop stats from {len(specs)} samples: "
            f"mean={self.dataset_mean:.4f}, std={self.dataset_std:.4f}"
        )
        return self

    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if self.dataset_mean is None or self.dataset_std is None:
            raise RuntimeError(
                "LogMelPadCrop has no normalization stats. "
                "Call .fit() or pass dataset_mean and dataset_std to __init__."
            )
        spec = self._to_raw_db(waveform, sr)
        return (spec - self.dataset_mean) / (self.dataset_std * 2.0)


class FolderAudioPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, files: list[str], transform: LogMelPadCrop):
        self.files = self._filter_decodable_files(files)
        self.transform = transform
        if not self.files:
            raise RuntimeError("No decodable audio files found for pretraining.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        waveform, sr = torchaudio.load(self.files[idx])
        spec = self.transform(waveform, sr)
        return {"x": spec}

    @staticmethod
    def _filter_decodable_files(files: list[str]) -> list[str]:
        valid_files = []
        skipped = 0
        for path in files:
            try:
                waveform, _ = torchaudio.load(path)
                if waveform.numel() == 0:
                    skipped += 1
                    continue
                valid_files.append(path)
            except Exception:
                skipped += 1
        if skipped:
            print(f"Skipped {skipped} unreadable audio files during dataset scan.")
        return valid_files


class WaveformArrayPretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        waveforms: np.ndarray,
        source_rate: int,
        transform: LogMelPadCrop,
    ):
        self.waveforms = waveforms
        self.source_rate = source_rate
        self.transform = transform

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int):
        waveform = torch.tensor(self.waveforms[idx], dtype=torch.float32)
        spec = self.transform(waveform, self.source_rate)
        return {"x": spec}


class WaveformArrayClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        waveforms: np.ndarray,
        labels: np.ndarray,
        source_rate: int,
        transform: LogMelPadCrop,
    ):
        self.waveforms = waveforms
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.source_rate = source_rate
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        waveform = torch.tensor(self.waveforms[idx], dtype=torch.float32)
        spec = self.transform(waveform, self.source_rate)
        return {"x": spec, "labels": self.labels[idx]}


class AudioASTClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        n_classes: int,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, n_classes)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x=None, labels=None):
        outputs = self.backbone(x)
        hidden = extract_last_hidden_state(outputs)
        if hidden.dim() == 3:
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden
        logits = self.head(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


def build_audio_classifier_from_backbone(
    backbone: nn.Module,
    n_classes: int,
    class_weights: torch.Tensor | None = None,
) -> AudioASTClassifier:
    hidden_size = int(getattr(backbone.config, "hidden_size"))
    return AudioASTClassifier(
        backbone=copy.deepcopy(backbone),
        hidden_size=hidden_size,
        n_classes=n_classes,
        class_weights=class_weights,
    )
