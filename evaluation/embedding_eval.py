#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from evaluation.visualization import plot_class_groups
from models.audio_common import extract_last_hidden_state


def _embedding_collator(batch, input_key: str):
    xs = []
    for item in batch:
        if isinstance(item, dict):
            xs.append(item[input_key])
        else:
            xs.append(item)
    return torch.stack(xs)


def compute_embeddings(backbone, dataset, batch_size: int = 32, input_key: str = "x"):
    backbone.eval()
    device = next(backbone.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: _embedding_collator(batch, input_key=input_key),
    )
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, dtype=torch.float32)
            outputs = backbone(batch)
            hidden = extract_last_hidden_state(outputs)
            if hidden.dim() == 3:
                hidden = hidden.mean(dim=1)
            embeddings.append(hidden.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def run_knn_eval(emb_train, y_train, emb_test, y_test) -> float:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(emb_train, y_train)
    preds = knn.predict(emb_test)
    return float((preds == y_test).mean())


def run_tsne(embeddings):
    if len(embeddings) < 2:
        return np.zeros((len(embeddings), 2), dtype=np.float32)
    perplexity = min(30, max(1, len(embeddings) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        init="random",
        random_state=42,
    )
    return tsne.fit_transform(embeddings)


def run_pca(embeddings):
    if len(embeddings) < 2:
        return np.zeros((len(embeddings), 2), dtype=np.float32)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(embeddings)


def plot_embeddings(reduced_embeddings, labels, method, output_dir, idx2cls=None):
    plot_class_groups(
        embeddings=reduced_embeddings,
        labels=labels,
        method=method,
        output_dir=output_dir,
        idx2cls=idx2cls,
    )


def evaluate_embedding_snapshots(
    backbone,
    train_dataset,
    y_train,
    test_dataset,
    y_test,
    output_dir: str | Path,
    idx2cls: dict[int, str],
    batch_size: int = 32,
    input_key: str = "x",
):
    epoch_dir = Path(output_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    emb_train = compute_embeddings(
        backbone,
        train_dataset,
        batch_size=batch_size,
        input_key=input_key,
    )
    emb_test = compute_embeddings(
        backbone,
        test_dataset,
        batch_size=batch_size,
        input_key=input_key,
    )

    knn_acc = run_knn_eval(emb_train, y_train, emb_test, y_test)
    (epoch_dir / "knn_results.txt").write_text(f"knn_accuracy: {knn_acc:.6f}\n")

    tsne_embeddings = run_tsne(emb_test)
    plot_embeddings(
        tsne_embeddings,
        y_test,
        method="tsne",
        output_dir=epoch_dir / "tsne",
        idx2cls=idx2cls,
    )

    pca_embeddings = run_pca(emb_test)
    plot_embeddings(
        pca_embeddings,
        y_test,
        method="pca",
        output_dir=epoch_dir / "pca",
        idx2cls=idx2cls,
    )

    return {
        "knn_accuracy": knn_acc,
        "emb_train": emb_train,
        "emb_test": emb_test,
    }
