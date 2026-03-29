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
            "ep45_knn":      _f(r.get("milestone_45_knn_acc", "")),
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
# Scratch and Pretrained only show initial KNN (their epoch-15/30 milestones
# are measured during fine-tuning, not pretraining).
# SSL methods show all available pretraining checkpoints: 0, 15, 30, 45.

SSL_KNN_EXPS = [
    ("MAE\n(AudioSet)", "MAE (AudioSet+ESC50)"),
    ("MAE\n(ESC50-only)", "MAE (ESC50-only)"),
    ("DINO",             "DINO (AudioSet+ESC50)"),
]
REF_EXPS = [
    ("Pretrained\nAST", "Pretrained AST"),
    ("Scratch\nAST",    "Scratch AST"),
]

# Epochs to display for SSL methods
EPOCHS_SSL = [0, 15, 30, 45]
EPOCH_KEYS  = ["initial_knn", "ep15_knn", "ep30_knn", "ep45_knn"]
EPOCH_ALPHAS = [0.35, 0.60, 0.85, 1.0]
EPOCH_HATCHES = [None, None, "//", "xx"]
EPOCH_LABELS  = ["Epoch 0 (init)", "Epoch 15", "Epoch 30", "Epoch 45"]

# ── layout: SSL groups on left, ref groups on right, separated by a gap ──
n_ssl = len(SSL_KNN_EXPS)
n_ref = len(REF_EXPS)
w = 0.18                         # bar width
group_w = 4 * w                  # width of one SSL group (4 bars)
ref_group_w = w + 0.04           # width of one ref group (1 bar + padding)
gap = 0.35

ssl_centers = np.array([i * (group_w + 0.1) for i in range(n_ssl)])
ref_start   = ssl_centers[-1] + group_w / 2 + gap + ref_group_w / 2
ref_centers = np.array([ref_start + i * (ref_group_w + 0.05) for i in range(n_ref)])

fig, ax = plt.subplots(figsize=(13, 5))

# ── draw SSL bars ──────────────────────────────────────────────────────────
for gi, (short, name) in enumerate(SSL_KNN_EXPS):
    r = by_name.get(name)
    if r is None:
        continue
    c = color(r["method"])
    cx = ssl_centers[gi]
    offsets = [cx + (j - 1.5) * w for j in range(4)]
    for j, (key, alpha, hatch, ep_label) in enumerate(
            zip(EPOCH_KEYS, EPOCH_ALPHAS, EPOCH_HATCHES, EPOCH_LABELS)):
        v = r.get(key)
        if v is None:
            continue
        bar = ax.bar(offsets[j], v, width=w, color=c, alpha=alpha,
                     hatch=hatch, edgecolor="white", linewidth=0.7)
        ax.text(offsets[j], v + 0.004, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7, color="black")

# group x-tick label centred under group
ax.set_xticks(list(ssl_centers) + list(ref_centers))
ax.set_xticklabels([s for s, _ in SSL_KNN_EXPS] + [s for s, _ in REF_EXPS], fontsize=9)

# ── draw reference (scratch / pretrained) bars — initial only ─────────────
for gi, (short, name) in enumerate(REF_EXPS):
    r = by_name.get(name)
    if r is None:
        continue
    v = r.get("initial_knn")
    if v is None:
        continue
    c = color(r["method"])
    cx = ref_centers[gi]
    ax.bar(cx, v, width=w + 0.04, color=c, alpha=0.35,
           edgecolor="white", linewidth=0.7)
    ax.text(cx, v + 0.004, f"{v:.2f}", ha="center", va="bottom",
            fontsize=7, color="black")

# ── legend ─────────────────────────────────────────────────────────────────
epoch_patches = [
    mpatches.Patch(color="#777", alpha=a, hatch=h, label=lbl)
    for a, h, lbl in zip(EPOCH_ALPHAS, EPOCH_HATCHES, EPOCH_LABELS)
]
method_patches = [
    mpatches.Patch(color=v, label=k)
    for k, v in METHOD_COLOR.items()
    if k in {r["method"] for r in rows}
]
ax.legend(handles=epoch_patches + method_patches,
          fontsize=8, loc="upper left", ncol=2, framealpha=0.85)

ax.set_ylabel("5-NN KNN Accuracy")
ax.set_title("KNN Representation Quality During Pretraining\n"
             "(Scratch / Pretrained: initial value only)")
fig.tight_layout()
fig.savefig(OUT_DIR / "chart3_knn_accuracy.png")
plt.close(fig)
print("Saved chart3_knn_accuracy.png")

# ── CHART 4: label efficiency across 25% / 50% / 100% ────────────────────────
LABEL_SERIES = {
    "MAE":     [("MAE (25% labels)", 25), ("MAE (50% labels)", 50), ("MAE (AudioSet+ESC50)", 100)],
    "Scratch": [("Scratch (25% labels)", 25), ("Scratch (50% labels)", 50), ("Scratch AST", 100)],
}

fig, ax = plt.subplots(figsize=(7, 4.5))
for method, entries in LABEL_SERIES.items():
    xs, ys = [], []
    for name, pct in entries:
        r = by_name.get(name)
        if r is None:
            continue
        acc = best_test(r)
        if acc is None:
            continue
        xs.append(pct)
        ys.append(acc)
    if not xs:
        continue
    c = color(method)
    ax.plot(xs, ys, marker="o", lw=2.2, color=c, label=method)
    for x_pt, y_pt in zip(xs, ys):
        ax.annotate(f"{y_pt:.3f}", xy=(x_pt, y_pt),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=9, color=c)

ax.set_xticks([25, 50, 100])
ax.set_xticklabels(["25%", "50%", "100%"])
ax.set_xlabel("Fraction of Training Labels Used")
ax.set_ylabel("Test Accuracy")
ax.set_title("Label Efficiency: Test Accuracy vs Training Label Fraction")
ax.set_ylim(0, 0.65)
ax.legend(fontsize=10, framealpha=0.85)
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

    # Left: final classification accuracy only
    ax = axes[0]
    final_accs = [r["final_test"] or 0 for r in src_rows]
    bars2 = ax.bar(x, final_accs, width=0.45, color=METHOD_COLOR["MAE"], edgecolor="white")
    for bar, v in zip(bars2, final_accs):
        if v:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(src_labels)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Final Classification Accuracy")
    ax.set_ylim(0, max(final_accs) * 1.3)

    # Right: KNN over pretraining epochs as a line chart
    ax = axes[1]
    colors_src = [METHOD_COLOR["MAE"], "#C44E52"]
    for i, (r, lbl, clr) in enumerate(zip(src_rows, src_labels, colors_src)):
        pts = [(0, r["initial_knn"]), (15, r["ep15_knn"]), (30, r["ep30_knn"])]
        xs = [ep for ep, v in pts if v is not None]
        ys = [v  for ep, v in pts if v is not None]
        ax.plot(xs, ys, marker="o", lw=2.2, color=clr, label=lbl)
        for xp, yp in zip(xs, ys):
            ax.annotate(f"{yp:.3f}", xy=(xp, yp), xytext=(0, 7),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=clr)
    ax.set_xticks([0, 15, 30])
    ax.set_xticklabels(["Epoch 0", "Epoch 15", "Epoch 30"])
    ax.set_ylabel("5-NN KNN Accuracy")
    ax.set_title("KNN Quality During Pretraining")
    ax.legend(fontsize=9)

    fig.suptitle("MAE Pre-training Data: AudioSet+ESC50 vs ESC50-only", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart5_mae_data_source.png")
    plt.close(fig)
    print("Saved chart5_mae_data_source.png")
else:
    print("Skipping chart5: MAE ESC50-only data not available.")

# ── CHART 7: MAE preprocessing ablation ──────────────────────────────────────
# Compare mae_big (no preprocessing) vs mae_big_preprocess (correct std)
import json as _json

mae_ablation = {
    "mae_big":            ("No Preprocessing\n(paper std ~4 dB)",  False),
    "mae_big_preprocess": ("Correct Preprocessing\n(actual std ~12 dB)", True),
}

mae_abl_data = {}
for folder, (label, preprocessed) in mae_ablation.items():
    p = Path("data/runs") / folder / "metrics.json"
    if p.exists():
        m = _json.loads(p.read_text())
        mae_abl_data[label] = {
            "preprocessed": preprocessed,
            "initial_knn":  m.get("initial_knn_acc"),
            "ep15_knn":     m.get("milestones", {}).get("15", {}).get("knn_acc"),
            "ep30_knn":     m.get("milestones", {}).get("30", {}).get("knn_acc"),
            "test_acc":     m.get("final", {}).get("test_acc"),
        }

if len(mae_abl_data) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels_abl = list(mae_abl_data.keys())
    clrs_abl   = [METHOD_COLOR["MAE"]] * 2
    x2 = np.arange(2)
    w2 = 0.28

    # Left: KNN progression
    ax = axes[0]
    init_v  = [mae_abl_data[l]["initial_knn"] or 0 for l in labels_abl]
    ep15_v  = [mae_abl_data[l]["ep15_knn"] or 0    for l in labels_abl]
    ep30_v  = [mae_abl_data[l]["ep30_knn"] or 0    for l in labels_abl]
    b1 = ax.bar(x2 - w2, init_v,  width=w2, color=METHOD_COLOR["MAE"], alpha=0.35, label="Initial KNN")
    b2 = ax.bar(x2,      ep15_v,  width=w2, color=METHOD_COLOR["MAE"],              label="Epoch-15 KNN")
    b3 = ax.bar(x2 + w2, ep30_v,  width=w2, color=METHOD_COLOR["MAE"], alpha=0.6,
                hatch="//", label="Epoch-30 KNN")
    for bars in [b1, b2, b3]:
        for bar in bars:
            v = bar.get_height()
            if v:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                        f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x2); ax.set_xticklabels(labels_abl, fontsize=9)
    ax.set_ylabel("5-NN KNN Accuracy")
    ax.set_title("KNN Representation Quality")
    ax.legend(fontsize=9)

    # Right: Final test accuracy
    ax = axes[1]
    test_v = [mae_abl_data[l]["test_acc"] or 0 for l in labels_abl]
    bars2  = ax.bar(x2, test_v, width=0.45, color=clrs_abl, edgecolor="white")
    for bar, v in zip(bars2, test_v):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x2); ax.set_xticklabels(labels_abl, fontsize=9)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Final Test Accuracy")
    ax.set_ylim(0, max(test_v) * 1.25)

    fig.suptitle("MAE: Effect of Data Normalization Correction", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart7_mae_preprocessing.png")
    plt.close(fig)
    print("Saved chart7_mae_preprocessing.png")

# ── CHART 8: DINO mean-feature-std (collapse monitoring) ──────────────────────
def parse_collapse(folder):
    p = Path("data/runs") / folder / "pretrain" / "collapse_metrics.txt"
    if not p.exists():
        return [], []
    epochs, stds = [], []
    seen = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = dict(item.split("=") for item in line.split(",") if "=" in item)
        ep = int(parts["epoch"])
        std = float(parts["mean_feature_std"])
        if ep not in seen:
            seen.add(ep)
            epochs.append(ep)
            stds.append(std)
    return epochs, stds

dino_collapse_runs = [
    # Round 1 (batch=8)
    ("dino_exp3_asymmetric",        "Asym. aug, batch=8\n(collapsed)",       "#DD8452", "--"),
    ("dino_big_preprocess",         "Soft aug, batch=8, out=256",            "#4C72B0", "-"),
    ("dino_exp1_outdim1024",        "Soft aug, batch=8, out=1024",           "#C44E52", "-"),
    ("dino_exp2_temp007",           "Soft aug, batch=8, out=1024, t=0.07",   "#8172B2", "-"),
    # Round 2 (batch=64 via gradient accumulation)
    ("dino_r2_exp1_accum",          "Soft aug, batch=64, out=256",           "#4C72B0", "-."),
    ("dino_r2_exp2_accum_1024_t07", "Soft aug, batch=64, out=1024, t=0.07",  "#8172B2", "-."),
    ("dino_r2_exp3_accum_asymmetric","Asym. aug, batch=64, out=1024",        "#2ca02c", "-."),
]

fig, ax = plt.subplots(figsize=(9, 5))
for folder, label, clr, ls in dino_collapse_runs:
    eps, stds = parse_collapse(folder)
    if not eps:
        continue
    eps_clip  = [e for e in eps  if e <= 30]
    stds_clip = [s for e, s in zip(eps, stds) if e <= 30]
    if not eps_clip:
        continue
    ax.plot(eps_clip, stds_clip, label=label, color=clr, lw=2.2, ls=ls)
    ax.annotate(f"{stds_clip[-1]:.3f}", xy=(eps_clip[-1], stds_clip[-1]),
                xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=clr)

ax.axhline(0.5, color="gray", lw=1, ls=":", label="Collapse threshold ≈ 0.5")
ax.set_xlabel("Pretraining Epoch")
ax.set_ylabel("Mean Feature Std")
ax.set_title("DINO Representation Collapse Monitoring\n(Higher = more diverse representations)")
ax.legend(fontsize=9, framealpha=0.85)
fig.tight_layout()
fig.savefig(OUT_DIR / "chart8_dino_collapse.png")
plt.close(fig)
print("Saved chart8_dino_collapse.png")

print("\nAll charts written to", OUT_DIR)
