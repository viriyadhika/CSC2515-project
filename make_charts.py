#!/usr/bin/env python3
"""Generate summary charts from data/runs/summary/audio_ast_results.csv."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CSV_PATH = Path("data/runs/summary/audio_ast_results.csv")
OUT_DIR  = Path("data/runs/summary")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

METHOD_COLOR = {
    "MAE":        "#4C72B0",
    "DINO":       "#DD8452",
    "Pretrained": "#C44E52",
    "Scratch":    "#8172B2",
}

def color(method):
    return METHOD_COLOR.get(method, "#999999")

def _f(v):
    return float(v) if v and v.strip() else None

rows = []
with open(CSV_PATH) as fh:
    for r in csv.DictReader(fh):
        rows.append({
            "experiment":    r["experiment"],
            "method":        r["method"],
            "pretrain_ep":   int(r["pretrain_epochs"]),
            "finetune_ep":   int(r["finetune_epochs"]),
            "pct":           int(r["percent_train"]),
            "initial_knn":   _f(r["initial_knn_acc"]),
            "ep15_knn":      _f(r["milestone_15_knn_acc"]),
            "ep30_knn":      _f(r["milestone_30_knn_acc"]),
            "ep15_test":     _f(r["milestone_15_test_acc"]),
            "ep30_test":     _f(r["milestone_30_test_acc"]),
            "final_test":    _f(r["final_test_acc"]),
            "final_test_f1": _f(r["final_test_f1"]),
        })

by_name = {r["experiment"]: r for r in rows}

def best_test(r):
    for v in [r["final_test"], r["ep30_test"], r["ep15_test"]]:
        if v is not None:
            return v
    return None

# ── CHART 1: main accuracy comparison (100% labelled data) ───────────────────
MAIN_EXPS = [
    ("Pretrained\nAST",    "Pretrained AST"),
    ("MAE\n(AudioSet)",    "MAE (AudioSet+ESC50)"),
    ("MAE\n(ESC50-only)",  "MAE (ESC50-only)"),
    ("DINO",               "DINO (AudioSet+ESC50)"),
    ("Scratch\nAST",       "Scratch AST"),
]

fig, ax = plt.subplots(figsize=(10, 5))
labels, accs, clrs = [], [], []
for short, name in MAIN_EXPS:
    r = by_name.get(name)
    if r is None:
        continue
    acc = best_test(r)
    if acc is None:
        continue
    labels.append(short)
    accs.append(acc)
    clrs.append(color(r["method"]))

x = np.arange(len(labels))
bars = ax.bar(x, accs, color=clrs, width=0.55, edgecolor="white", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Test Accuracy")
ax.set_title("Model Comparison – Final Test Accuracy (100% labelled data)")
ax.set_ylim(0, max(accs) * 1.2 if accs else 1.0)
ax.axhline(0, color="black", linewidth=0.6)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.005,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in METHOD_COLOR.items()
                  if k in {r["method"] for r in rows}]
ax.legend(handles=legend_patches, loc="upper right", framealpha=0.85)
fig.tight_layout()
fig.savefig(OUT_DIR / "chart1_model_comparison.png")
plt.close(fig)
print("Saved chart1_model_comparison.png")

# ── CHART 2: accuracy at milestones ──────────────────────────────────────────
PROG_EXPS = [
    ("Pretrained AST", "Pretrained AST"),
    ("MAE (AudioSet)", "MAE (AudioSet+ESC50)"),
    ("MAE (ESC50-only)", "MAE (ESC50-only)"),
    ("DINO",           "DINO (AudioSet+ESC50)"),
    ("Scratch AST",    "Scratch AST"),
]

fig, ax = plt.subplots(figsize=(9, 5))
for short, name in PROG_EXPS:
    r = by_name.get(name)
    if r is None:
        continue
    pts = [r["ep15_test"], r["ep30_test"], r["final_test"]]
    xs = [i for i, v in enumerate(pts) if v is not None]
    ys = [v for v in pts if v is not None]
    if len(xs) < 1:
        continue
    c = color(r["method"])
    ax.plot(xs, ys, marker="o", lw=2, color=c, label=short)
    for xi, yi in zip(xs, ys):
        ax.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", fontsize=8, color=c)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Epoch 15", "Epoch 30", "Final"])
ax.set_ylabel("Test Accuracy")
ax.set_title("Test Accuracy at Training Milestones")
ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
fig.tight_layout()
fig.savefig(OUT_DIR / "chart2_accuracy_progression.png")
plt.close(fig)
print("Saved chart2_accuracy_progression.png")

# ── CHART 3: KNN quality at milestones ───────────────────────────────────────
KNN_EXPS = [
    ("Pretrained AST", "Pretrained AST"),
    ("MAE (AudioSet)", "MAE (AudioSet+ESC50)"),
    ("MAE (ESC50-only)", "MAE (ESC50-only)"),
    ("DINO",           "DINO (AudioSet+ESC50)"),
    ("Scratch AST",    "Scratch AST"),
]

knn_shorts, knn_init_v, knn_ep15_v, knn_ep30_v, knn_clrs = [], [], [], [], []
for short, name in KNN_EXPS:
    r = by_name.get(name)
    if r is None:
        continue
    if all(v is None for v in [r["initial_knn"], r["ep15_knn"], r["ep30_knn"]]):
        continue
    knn_shorts.append(short)
    knn_init_v.append(r["initial_knn"])
    knn_ep15_v.append(r["ep15_knn"])
    knn_ep30_v.append(r["ep30_knn"])
    knn_clrs.append(color(r["method"]))

n = len(knn_shorts)
x = np.arange(n)
w = 0.25
fig, ax = plt.subplots(figsize=(11, 5))

def _bar(offsets, vals, label, alpha=1.0, hatch=None):
    for xi, v, c in zip(offsets, vals, knn_clrs):
        if v is None:
            continue
        b = ax.bar(xi, v, width=w, color=c, alpha=alpha,
                   hatch=hatch, edgecolor="white", linewidth=0.7)
        ax.text(xi, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

_bar(x - w, knn_init_v, "Initial", alpha=0.35)
_bar(x,     knn_ep15_v, "Epoch 15")
_bar(x + w, knn_ep30_v, "Epoch 30", alpha=0.7, hatch="//")

ax.set_xticks(x)
ax.set_xticklabels(knn_shorts, fontsize=9)
ax.set_ylabel("5-NN KNN Accuracy")
ax.set_title("KNN Representation Quality at Training Milestones")
legend = [
    mpatches.Patch(color="#777", alpha=0.35, label="Initial (before training)"),
    mpatches.Patch(color="#777", label="After epoch 15"),
    mpatches.Patch(color="#777", alpha=0.7, hatch="//", label="After epoch 30"),
] + [mpatches.Patch(color=v, label=k) for k, v in METHOD_COLOR.items()
     if k in {r["method"] for r in rows}]
ax.legend(handles=legend, fontsize=8.5, loc="upper right", ncol=2, framealpha=0.85)
fig.tight_layout()
fig.savefig(OUT_DIR / "chart3_knn_accuracy.png")
plt.close(fig)
print("Saved chart3_knn_accuracy.png")

# ── CHART 4: data ablation (100% vs 25% labels) ──────────────────────────────
ABLATION = [
    ("MAE\n(100%)",    "MAE (AudioSet+ESC50)"),
    ("MAE\n(25%)",     "MAE (25% labels)"),
    ("Scratch\n(100%)", "Scratch AST"),
    ("Scratch\n(25%)",  "Scratch (25% labels)"),
]

fig, ax = plt.subplots(figsize=(7, 4.5))
ab_labels, ab_accs, ab_clrs, ab_hatches = [], [], [], []
for short, name in ABLATION:
    r = by_name.get(name)
    if r is None:
        continue
    acc = best_test(r)
    if acc is None:
        continue
    ab_labels.append(short)
    ab_accs.append(acc)
    ab_clrs.append(color(r["method"]))
    ab_hatches.append("" if r["pct"] == 100 else "//")

x = np.arange(len(ab_labels))
bars = ax.bar(x, ab_accs, color=ab_clrs, hatch=ab_hatches,
              width=0.5, edgecolor="white", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(ab_labels, fontsize=10)
ax.set_ylabel("Test Accuracy")
ax.set_title("Label Efficiency: 100% vs 25% Training Data")
ylim = max(ab_accs) * 1.25 if ab_accs else 0.5
ax.set_ylim(0, ylim)
for bar, acc in zip(bars, ab_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.005,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=10)
legend_patches = [
    mpatches.Patch(color="#777", label="100% data"),
    mpatches.Patch(color="#777", hatch="//", label="25% data"),
] + [mpatches.Patch(color=v, label=k) for k, v in METHOD_COLOR.items()
     if k in {"MAE", "Scratch"}]
ax.legend(handles=legend_patches, fontsize=9, framealpha=0.85)
fig.tight_layout()
fig.savefig(OUT_DIR / "chart4_data_ablation.png")
plt.close(fig)
print("Saved chart4_data_ablation.png")

# ── CHART 5: MAE data source ablation ────────────────────────────────────────
src_names = ["MAE (AudioSet+ESC50)", "MAE (ESC50-only)"]
src_labels = ["AudioSet+ESC50", "ESC50-only"]
src_rows = [by_name.get(n) for n in src_names]

if all(r is not None for r in src_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(2)
    w = 0.35

    # Left: classification accuracy
    ax = axes[0]
    ep15_accs = [r["ep15_test"] or 0 for r in src_rows]
    final_accs = [r["final_test"] or 0 for r in src_rows]
    b1 = ax.bar(x - w/2, ep15_accs, width=w, color=METHOD_COLOR["MAE"], label="Epoch-15")
    b2 = ax.bar(x + w/2, final_accs, width=w, color=METHOD_COLOR["MAE"],
                alpha=0.55, hatch="//", label="Final")
    for bars in [b1, b2]:
        for bar in bars:
            v = bar.get_height()
            if v:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                        f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(src_labels)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Classification Accuracy")
    ax.legend(fontsize=9)

    # Right: KNN accuracy
    ax = axes[1]
    init_knns = [r["initial_knn"] or 0 for r in src_rows]
    ep15_knns = [r["ep15_knn"] or 0 for r in src_rows]
    ep30_knns = [r["ep30_knn"] or 0 for r in src_rows]
    b1 = ax.bar(x - w, init_knns, width=w, color=METHOD_COLOR["MAE"], alpha=0.35, label="Initial KNN")
    b2 = ax.bar(x,     ep15_knns, width=w, color=METHOD_COLOR["MAE"], label="Epoch-15 KNN")
    b3 = ax.bar(x + w, ep30_knns, width=w, color=METHOD_COLOR["MAE"],
                alpha=0.6, hatch="//", label="Epoch-30 KNN")
    for bars in [b1, b2, b3]:
        for bar in bars:
            v = bar.get_height()
            if v:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                        f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(src_labels)
    ax.set_ylabel("5-NN KNN Accuracy")
    ax.set_title("KNN Representation Quality")
    ax.legend(fontsize=9)

    fig.suptitle("MAE Pre-training Data: AudioSet+ESC50 vs ESC50-only", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart5_mae_data_source.png")
    plt.close(fig)
    print("Saved chart5_mae_data_source.png")
else:
    print("Skipping chart5: MAE ESC50-only data not available.")

print("\nAll charts written to", OUT_DIR)
