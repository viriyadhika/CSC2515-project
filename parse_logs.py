#!/usr/bin/env python3
"""
Parse metrics.json files from each experiment run and write summary CSVs.

Each experiment's output_dir must contain:
  - metrics.json   (written by the training scripts)

Outputs:
  data/runs/summary/audio_ast_results.csv
  data/runs/summary/audio_ast_key_results.csv
"""

import csv
import json
from pathlib import Path

BASE = Path("data/runs")
SUMMARY = BASE / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = [
    {
        "experiment": "MAE (AudioSet+ESC50)",
        "method": "MAE",
        "pretrain_epochs": 30,
        "finetune_epochs": 45,
        "pretrain_data": "AudioSet+ESC50",
        "percent_train": 100,
        "output_dir": "data/runs/mae_big_preprocess",
    },
    {
        "experiment": "MAE (ESC50-only)",
        "method": "MAE",
        "pretrain_epochs": 30,
        "finetune_epochs": 45,
        "pretrain_data": "ESC50-only",
        "percent_train": 100,
        "output_dir": "data/runs/mae_esc50_preprocess",
    },
    {
        "experiment": "DINO (AudioSet+ESC50)",
        "method": "DINO",
        "pretrain_epochs": 30,
        "finetune_epochs": 45,
        "pretrain_data": "AudioSet+ESC50",
        "percent_train": 100,
        "output_dir": "data/runs/dino",
    },
    {
        "experiment": "Pretrained AST",
        "method": "Pretrained",
        "pretrain_epochs": 0,
        "finetune_epochs": 45,
        "pretrain_data": "HF weights",
        "percent_train": 100,
        "output_dir": "data/runs/pretrained",
    },
    {
        "experiment": "Scratch AST",
        "method": "Scratch",
        "pretrain_epochs": 0,
        "finetune_epochs": 45,
        "pretrain_data": "N/A",
        "percent_train": 100,
        "output_dir": "data/runs/scratch",
    },
    {
        "experiment": "MAE (25% labels)",
        "method": "MAE",
        "pretrain_epochs": 30,
        "finetune_epochs": 45,
        "pretrain_data": "AudioSet+ESC50",
        "percent_train": 25,
        "output_dir": "data/runs/mae_25_pct",
    },
    {
        "experiment": "MAE (50% labels)",
        "method": "MAE",
        "pretrain_epochs": 30,
        "finetune_epochs": 45,
        "pretrain_data": "AudioSet+ESC50",
        "percent_train": 50,
        "output_dir": "data/runs/mae_50_pct",
    },
    {
        "experiment": "Scratch (25% labels)",
        "method": "Scratch",
        "pretrain_epochs": 0,
        "finetune_epochs": 45,
        "pretrain_data": "N/A",
        "percent_train": 25,
        "output_dir": "data/runs/scratch_pct25",
    },
    {
        "experiment": "Scratch (50% labels)",
        "method": "Scratch",
        "pretrain_epochs": 0,
        "finetune_epochs": 45,
        "pretrain_data": "N/A",
        "percent_train": 50,
        "output_dir": "data/runs/scratch_50_pct",
    },
    {
        "experiment": "DINO (batch=64, out=256)",
        "method": "DINO",
        "pretrain_epochs": 30,
        "finetune_epochs": 45,
        "pretrain_data": "AudioSet+ESC50",
        "percent_train": 100,
        "output_dir": "data/runs/dino_r2_exp1_accum",
    },
    {
        "experiment": "DINO (batch=64, out=1024, temp=0.07)",
        "method": "DINO",
        "pretrain_epochs": 30,
        "finetune_epochs": 45,
        "pretrain_data": "AudioSet+ESC50",
        "percent_train": 100,
        "output_dir": "data/runs/dino_r2_exp2_accum_1024_t07",
    },
    {
        "experiment": "DINO (batch=64, asymmetric aug)",
        "method": "DINO",
        "pretrain_epochs": 30,
        "finetune_epochs": 45,
        "pretrain_data": "AudioSet+ESC50",
        "percent_train": 100,
        "output_dir": "data/runs/dino_r2_exp3_accum_asymmetric",
    },
]


META_COLS = [
    "experiment",
    "method",
    "pretrain_epochs",
    "finetune_epochs",
    "pretrain_data",
    "percent_train",
]

METRIC_COLS = [
    "initial_knn_acc",
    "milestone_15_knn_acc",
    "milestone_30_knn_acc",
    "milestone_45_knn_acc",
    "milestone_15_val_acc",
    "milestone_15_val_f1",
    "milestone_15_test_acc",
    "milestone_15_test_f1",
    "milestone_30_val_acc",
    "milestone_30_val_f1",
    "milestone_30_test_acc",
    "milestone_30_test_f1",
    "final_val_acc",
    "final_val_f1",
    "final_test_acc",
    "final_test_f1",
]


def _fmt(v):
    return f"{v:.4f}" if v is not None else ""


def load_experiment_metrics(output_dir: str) -> dict:
    path = Path(output_dir) / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def extract_row(exp: dict) -> dict:
    m = load_experiment_metrics(exp["output_dir"])
    row = {col: exp.get(col, "") for col in META_COLS}

    row["initial_knn_acc"] = _fmt(m.get("initial_knn_acc"))

    milestones = m.get("milestones", {})
    for ep in (15, 30, 45):
        ms = milestones.get(str(ep), {})
        row[f"milestone_{ep}_knn_acc"] = _fmt(ms.get("knn_acc"))
        if ep != 45:
            row[f"milestone_{ep}_val_acc"] = _fmt(ms.get("val_acc"))
            row[f"milestone_{ep}_val_f1"] = _fmt(ms.get("val_f1"))
            row[f"milestone_{ep}_test_acc"] = _fmt(ms.get("test_acc"))
            row[f"milestone_{ep}_test_f1"] = _fmt(ms.get("test_f1"))

    final = m.get("final", {})
    row["final_val_acc"] = _fmt(final.get("val_acc"))
    row["final_val_f1"] = _fmt(final.get("val_f1"))
    row["final_test_acc"] = _fmt(final.get("test_acc"))
    row["final_test_f1"] = _fmt(final.get("test_f1"))

    return row


rows = []
for exp in EXPERIMENTS:
    output_dir = exp["output_dir"]
    if not (Path(output_dir) / "metrics.json").exists():
        print(f"WARNING: {output_dir}/metrics.json not found, skipping.")
        continue
    rows.append(extract_row(exp))

all_cols = META_COLS + METRIC_COLS
out_path = SUMMARY / "audio_ast_results.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_cols)
    w.writeheader()
    w.writerows(rows)
print(f"Written {len(rows)} rows to {out_path}")

key_cols = [
    "experiment", "method", "pretrain_epochs", "finetune_epochs", "percent_train",
    "initial_knn_acc", "milestone_15_knn_acc", "milestone_30_knn_acc", "milestone_45_knn_acc",
    "milestone_15_test_acc", "milestone_30_test_acc",
    "final_test_acc", "final_test_f1",
]
compact_path = SUMMARY / "audio_ast_key_results.csv"
with open(compact_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=key_cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"Written {len(rows)} rows to {compact_path}")
