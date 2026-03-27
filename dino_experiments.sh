#!/usr/bin/env bash
# DINO ablation experiments — Round 2, designed to fit within 12h on an RTX 4070.
#
# Round 1 failures (all 3 experiments):
#   KNN never exceeded the random-init baseline (0.1175).
#   Root cause: effective batch size = 8. DINO's teacher EMA and center update
#   need large batches; with batch=8 the center converges to a single batch's
#   mean, making the teacher's softmax target noisy and near-constant.
#   Exp3 (asymmetric masking) collapsed completely (mean_feature_std=0.334).
#
# Fix applied in audio_ast_dino.py:
#   - default teacher_momentum lowered 0.996 → 0.99 (tuned for large batches)
#   - --gradient_accumulation_steps added (default=8 → effective batch = 64)
#
# Experiment 1 — Gradient accumulation only (baseline fix)
#   Verify that effective batch=64 + momentum=0.99 alone stops collapse.
#   Keep out_dim=256 to isolate the batch-size fix from head-size changes.
#
# Experiment 2 — Gradient accumulation + out_dim=1024 + temp=0.07
#   Best combination from Round 1 theory (head size + softer teacher),
#   now with the batch-size fix. Expected to be the strongest experiment.
#
# Experiment 3 — Gradient accumulation + asymmetric masking + out_dim=1024
#   Re-test DINOSR-inspired asymmetric masking now that the teacher is stable.
#   In Round 1 this collapsed; the masking itself is not the cause — the
#   noisy teacher was. With proper effective batch it may work.
#
# Timing (RTX 4070, effective_batch=64, ~0.6 it/s with accum=8):
#   30 pretrain epochs ≈ 2.9h | 30 finetune epochs ≈ 15min | Total 3 runs ≈ 9.6h
set -euo pipefail

ACCUM=8  # effective batch = pretrain_batch_size(8) * ACCUM(8) = 64

# ── Experiment 1: gradient accumulation fix (small head baseline) ─────────────
echo "=== DINO Exp1: grad_accum=8, out_dim=256 (batch-size fix baseline) ==="
python3 audio_ast_dino.py \
  --output_dir data/runs/dino_r2_exp1_accum \
  --out_dim 256 \
  --gradient_accumulation_steps $ACCUM \
  --pretrain_epochs 30 \
  --finetune_epochs 30

# ── Experiment 2: grad_accum + large head + softer temp ──────────────────────
echo "=== DINO Exp2: grad_accum=8, out_dim=1024, teacher_temp=0.07 ==="
python3 audio_ast_dino.py \
  --output_dir data/runs/dino_r2_exp2_accum_1024_t07 \
  --out_dim 1024 \
  --teacher_temp 0.07 \
  --gradient_accumulation_steps $ACCUM \
  --pretrain_epochs 30 \
  --finetune_epochs 30

# ── Experiment 3: grad_accum + asymmetric masking (re-test) ──────────────────
echo "=== DINO Exp3: grad_accum=8, out_dim=1024, asymmetric masking ==="
python3 audio_ast_dino.py \
  --output_dir data/runs/dino_r2_exp3_accum_asymmetric \
  --out_dim 1024 \
  --asymmetric_aug \
  --gradient_accumulation_steps $ACCUM \
  --pretrain_epochs 30 \
  --finetune_epochs 30

echo "=== DINO Round 2 experiments complete ==="
echo "Compare: python3 parse_logs.py && python3 make_charts.py"
