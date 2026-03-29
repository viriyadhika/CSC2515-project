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
    "MAE":              dict(color="#4C72B0", lw=2.2, ls="-"),
    "DINO":             dict(color="#DD8452", lw=2.2, ls="--"),
    "MAE ESC-50 Only":  dict(color="#55A868", lw=2.0, ls="-."),
    "Pretrained":       dict(color="#C44E52", lw=2.2, ls="-"),
    "Scratch":          dict(color="#8172B2", lw=2.2, ls="-"),
}

# Labels excluded from the "all models except pretrained" chart
EXCLUDE_FROM_MAIN = {"Pretrained"}

# Each entry: label → path to trainer_state.json
RUNS = {
    "MAE":        BASE / "mae_big_preprocess"        / "final_finetune" / "checkpoint-6750" / "trainer_state.json",
    "DINO":       BASE / "dino"       / "final_finetune" / "checkpoint-6750" / "trainer_state.json",
    "MAE ESC-50 Only":   BASE / "mae_esc50_preprocess"       / "final_finetune" / "checkpoint-6750" / "trainer_state.json",
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

# Combined side-by-side: left = all models except Pretrained, right = MAE vs Pretrained
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel — validation accuracy, Pretrained excluded
for label, ep_dict in data.items():
    if label in EXCLUDE_FROM_MAIN:
        continue
    sorted_eps = sorted(ep_dict.keys())
    ys = [ep_dict[ep].get("val_acc") for ep in sorted_eps]
    xs = [e for e, y in zip(sorted_eps, ys) if y is not None]
    ys = [y for y in ys if y is not None]
    if not xs:
        continue
    ax_left.plot(xs, ys, label=label, **STYLES.get(label, {}))

ax_left.set_title("Validation Accuracy (SSL & Scratch)")
ax_left.set_xlabel("Finetune Epoch")
ax_left.set_ylabel("Accuracy")
ax_left.legend(fontsize=9, framealpha=0.85)

# Right panel — MAE vs Pretrained only
for label in ("MAE", "Pretrained"):
    ep_dict = data.get(label)
    if ep_dict is None:
        continue
    sorted_eps = sorted(ep_dict.keys())
    ys = [ep_dict[ep].get("val_acc") for ep in sorted_eps]
    xs = [e for e, y in zip(sorted_eps, ys) if y is not None]
    ys = [y for y in ys if y is not None]
    if not xs:
        continue
    ax_right.plot(xs, ys, label=label, **STYLES.get(label, {}))

ax_right.set_title("Validation Accuracy: MAE vs Pretrained AST")
ax_right.set_xlabel("Finetune Epoch")
ax_right.set_ylabel("Accuracy")
ax_right.legend(fontsize=9, framealpha=0.85)

fig.suptitle("Fine-tuning Validation Accuracy", fontsize=13, fontweight="bold")
fig.tight_layout()
out = OUT_DIR / "chart6_val_acc_combined.png"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")

# Loss curves: MAE vs MAE ESC-50 Only vs Scratch (train loss + val loss)
LOSS_LABELS = ["MAE", "MAE ESC-50 Only", "Scratch"]

fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(12, 5))

for label in LOSS_LABELS:
    ep_dict = data.get(label)
    if ep_dict is None:
        continue
    sorted_eps = sorted(ep_dict.keys())
    style = STYLES.get(label, {})

    # Train loss
    ys_train = [ep_dict[ep].get("train_loss") for ep in sorted_eps]
    xs_t = [e for e, y in zip(sorted_eps, ys_train) if y is not None]
    ys_t = [y for y in ys_train if y is not None]
    if xs_t:
        ax_train.plot(xs_t, ys_t, label=label, **style)

    # Val loss
    ys_val = [ep_dict[ep].get("val_loss") for ep in sorted_eps]
    xs_v = [e for e, y in zip(sorted_eps, ys_val) if y is not None]
    ys_v = [y for y in ys_val if y is not None]
    if xs_v:
        ax_val.plot(xs_v, ys_v, label=label, **style)

ax_train.set_title("Training Loss")
ax_train.set_xlabel("Finetune Epoch")
ax_train.set_ylabel("Loss")
ax_train.legend(fontsize=9, framealpha=0.85)

ax_val.set_title("Validation Loss")
ax_val.set_xlabel("Finetune Epoch")
ax_val.set_ylabel("Loss")
ax_val.legend(fontsize=9, framealpha=0.85)

fig.suptitle("Fine-tuning Loss Curves: MAE vs. Scratch", fontsize=13, fontweight="bold")
fig.tight_layout()
out = OUT_DIR / "chart6b_loss_curves.png"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
