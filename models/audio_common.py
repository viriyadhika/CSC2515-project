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
        dataset_mean: float = -4.2677393,
        dataset_std: float = 4.5689974,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_length = target_length
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
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
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
            pad = self.target_length - t
            spec = torch.nn.functional.pad(spec, (0, 0, 0, pad))
        elif t > self.target_length:
            start = random.randint(0, t - self.target_length)
            spec = spec[start : start + self.target_length]

        spec = (spec - self.dataset_mean) / (self.dataset_std * 2.0)
        return spec


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
