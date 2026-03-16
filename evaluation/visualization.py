#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_class_groups(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str,
    output_dir: str | Path,
    idx2cls: dict[int, str] | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    x_min = float(np.min(embeddings[:, 0]))
    x_max = float(np.max(embeddings[:, 0]))
    y_min = float(np.min(embeddings[:, 1]))
    y_max = float(np.max(embeddings[:, 1]))

    max_label = int(np.max(labels)) if len(labels) else -1
    for start in range(0, max_label + 1, 5):
        end = min(start + 4, max_label)
        group = np.arange(start, end + 1)
        mask = np.isin(labels, group)

        fig, ax = plt.subplots(figsize=(6, 6))
        for class_id in group:
            class_mask = labels == class_id
            if not np.any(class_mask):
                continue
            label = idx2cls.get(class_id, str(class_id)) if idx2cls else str(class_id)
            ax.scatter(
                embeddings[class_mask, 0],
                embeddings[class_mask, 1],
                s=10,
                label=label,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{method.upper()} classes {start}-{end}")
        ax.legend(markerscale=2)
        fig.tight_layout()
        fig.savefig(output_path / f"{method.lower()}_classes_{start}_{end}.png")
        plt.close(fig)
