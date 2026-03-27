#!/usr/bin/env python3
"""
Plot per-epoch finetuning curves (train loss, val loss, val accuracy) for all
experiments. Reads from HuggingFace trainer_state.json files written during
training.

SSL runs:   {output_dir}/final_finetune/trainer_state.json
Supervised: {output_dir}/trainer/trainer_state.json
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("data/runs")
OUT_DIR = BASE / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

STYLES = {
    "MAE":        dict(color="#4C72B0", lw=2.2, ls="-"),
    "DINO":       dict(color="#DD8452", lw=2.2, ls="--"),
    "Pretrained": dict(color="#C44E52", lw=2.2, ls="-"),
    "Scratch":    dict(color="#8172B2", lw=2.2, ls="-"),
}

# Each entry: label → path to trainer_state.json
RUNS = {
    "MAE":        BASE / "mae_big_preprocess"        / "final_finetune" / "checkpoint-6750" / "trainer_state.json",
    "DINO":       BASE / "dino"       / "final_finetune" / "checkpoint-6750" / "trainer_state.json",
    "MAE ESC-50 Only":   BASE / "mae_esc50_only"       / "final_finetune" / "checkpoint-6750" / "trainer_state.json",
    "Pretrained": BASE / "pretrained" / "trainer"        / "trainer_state.json",
    "Scratch":    BASE / "scratch"    / "trainer"        / "trainer_state.json",
}


def parse_trainer_state(path: Path) -> dict[int, dict]:
    """
    Parse trainer_state.json and return {epoch -> {train_loss, val_loss, val_acc}}.
    HF Trainer logs two types of entries per epoch:
      - training:   {"loss": X, "epoch": N, ...}
      - evaluation: {"eval_loss": X, "eval_accuracy": Y, "epoch": N, ...}
    """
    data = json.loads(path.read_text())
    log_history = data.get("log_history", [])

    epochs: dict[int, dict] = {}
    for entry in log_history:
        raw_ep = entry.get("epoch")
        if raw_ep is None:
            continue
        ep = int(round(float(raw_ep)))
        if ep not in epochs:
            epochs[ep] = {}

        if "loss" in entry and "eval_loss" not in entry:
            epochs[ep]["train_loss"] = float(entry["loss"])
        if "eval_loss" in entry:
            epochs[ep]["val_loss"] = float(entry["eval_loss"])
            if "eval_accuracy" in entry:
                epochs[ep]["val_acc"] = float(entry["eval_accuracy"])

    return epochs


# Load data
data: dict[str, dict[int, dict]] = {}
for label, path in RUNS.items():
    if not path.exists():
        print(f"WARNING: {path} not found, skipping {label}.")
        continue
    data[label] = parse_trainer_state(path)

if not data:
    print("No data found. Run experiments first.")
    raise SystemExit(1)

# Combined 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ["Train Loss", "Validation Loss", "Validation Accuracy"]
metrics = ["train_loss", "val_loss", "val_acc"]

for ax, metric, title in zip(axes, metrics, titles):
    for label, ep_dict in data.items():
        sorted_eps = sorted(ep_dict.keys())
        ys = [ep_dict[ep].get(metric) for ep in sorted_eps]
        xs = [e for e, y in zip(sorted_eps, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        if not xs:
            continue
        style = STYLES.get(label, {})
        ax.plot(xs, ys, label=label, **style)

    ax.set_title(title)
    ax.set_xlabel("Finetune Epoch")
    ax.set_ylabel("Loss" if "loss" in metric else "Accuracy")

axes[0].legend(fontsize=9, loc="upper right", framealpha=0.85)
fig.suptitle("Finetuning Curves", fontsize=13, fontweight="bold")
fig.tight_layout()
out = OUT_DIR / "chart6_finetune_curves.png"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")

# Individual larger subplots
for metric, title, fname in [
    ("train_loss", "Train Loss during Finetuning",         "chart6a_train_loss.png"),
    ("val_loss",   "Validation Loss during Finetuning",    "chart6b_val_loss.png"),
    ("val_acc",    "Validation Accuracy during Finetuning", "chart6c_val_acc.png"),
]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ep_dict in data.items():
        sorted_eps = sorted(ep_dict.keys())
        ys = [ep_dict[ep].get(metric) for ep in sorted_eps]
        xs = [e for e, y in zip(sorted_eps, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        if not xs:
            continue
        style = STYLES.get(label, {})
        ax.plot(xs, ys, label=label, **style)

    ax.set_title(title)
    ax.set_xlabel("Finetune Epoch")
    ax.set_ylabel("Loss" if "loss" in metric else "Accuracy")
    ax.legend(fontsize=10, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"Saved {OUT_DIR / fname}")
