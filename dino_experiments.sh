#!/usr/bin/env bash
# DINO ablation experiments — designed to fit within 12h on an RTX 4070.
# Baseline (failed): out_dim=256, teacher_temp=0.04, symmetric aug → KNN dropped below random.
#
# Experiment 1 — Large head (DINO v1 fix)
#   Hypothesis: out_dim=256 is the primary collapse driver.
#   With only 256 dims and teacher_temp=0.04, softmax becomes near one-hot after
#   a single epoch, making it trivial to minimize loss without learning structure.
#   DINO v1 uses 65536; we use 1024 as a compute-affordable middle ground.
#
# Experiment 2 — Large head + softer teacher temperature (DINO v1 style)
#   Hypothesis: teacher_temp=0.04 is too aggressive on top of the small head.
#   DINO v1 warms teacher_temp from 0.04→0.07 over 30 epochs; we use 0.07 fixed.
#   Softer targets mean the student must match a broader distribution,
#   providing richer gradient signal and reducing the risk of mode collapse.
#
# Experiment 3 — Large head + asymmetric masking (DINOSR-inspired)
#   Hypothesis: symmetric weak augmentations don't force the model to learn
#   meaningful representations. DINOSR showed that masking 80% of frames on
#   the student view while the teacher sees the full input prevents collapse
#   without temperature tricks. Here student sees 3×(T/6) masked blocks (~50%),
#   teacher sees the clean spectrogram.
#
# Timing estimate (RTX 4070, ~5 it/s, 1716 steps/epoch with 16k AudioSet clips):
#   30 pretrain epochs ≈ 2.9h | 30 finetune epochs ≈ 15min | Total 3 runs ≈ 9.6h
set -euo pipefail

# ── Experiment 1: out_dim 256→1024 ───────────────────────────────────────────
echo "=== DINO Exp1: out_dim=1024 (large head) ==="
python3 audio_ast_dino.py \
  --output_dir data/runs/dino_exp1_outdim1024 \
  --out_dim 1024 \
  --pretrain_epochs 30 \
  --finetune_epochs 30

# ── Experiment 2: out_dim=1024 + softer teacher temperature ──────────────────
echo "=== DINO Exp2: out_dim=1024 + teacher_temp=0.07 ==="
python3 audio_ast_dino.py \
  --output_dir data/runs/dino_exp2_temp007 \
  --out_dim 1024 \
  --teacher_temp 0.07 \
  --pretrain_epochs 30 \
  --finetune_epochs 30

# ── Experiment 3: out_dim=1024 + asymmetric masking (DINOSR-inspired) ────────
echo "=== DINO Exp3: out_dim=1024 + asymmetric masking ==="
python3 audio_ast_dino.py \
  --output_dir data/runs/dino_exp3_asymmetric \
  --out_dim 1024 \
  --asymmetric_aug \
  --pretrain_epochs 30 \
  --finetune_epochs 30

echo "=== DINO experiments complete ==="
echo "Compare: python3 parse_logs.py && python3 make_charts.py"
