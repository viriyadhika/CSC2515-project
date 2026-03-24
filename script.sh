#!/usr/bin/env bash
# Standardised experiment suite: 30 epochs SSL pre-training + 45 epochs fine-tuning.
# All metrics are written to data/runs/<name>/metrics.json.
# Per-epoch training curves are in data/runs/<name>/final_finetune/trainer_state.json
# (SSL) or data/runs/<name>/trainer/trainer_state.json (supervised).
set -euo pipefail

# ── MAE: AudioSet + ESC50 pre-training ───────────────────────────────────────
echo "=== MAE (AudioSet+ESC50): 30 pretrain + 45 finetune ==="
python3 audio_ast_mae.py \
  --output_dir data/runs/mae \
  --pretrain_epochs 30 \
  --finetune_epochs 45

# ── MAE: ESC50-only pre-training (data-source ablation) ──────────────────────
echo "=== MAE (ESC50-only): 30 pretrain + 45 finetune ==="
python3 audio_ast_mae.py \
  --output_dir data/runs/mae_esc50_only \
  --pretrain_epochs 30 \
  --finetune_epochs 45 \
  --esc50_only_pretrain

# ── DINO: AudioSet + ESC50 pre-training ──────────────────────────────────────
echo "=== DINO (AudioSet+ESC50): 30 pretrain + 45 finetune ==="
python3 audio_ast_dino.py \
  --output_dir data/runs/dino \
  --pretrain_epochs 30 \
  --finetune_epochs 45

# ── Supervised baseline: pre-trained AST weights ─────────────────────────────
echo "=== Pretrained AST: 45 epochs supervised ==="
python3 audio_ast.py \
  --output_dir data/runs/pretrained \
  --epochs 45 \
  --finetune

# ── Supervised baseline: random initialisation (scratch) ─────────────────────
echo "=== Scratch AST: 45 epochs supervised ==="
python3 audio_ast.py \
  --output_dir data/runs/scratch \
  --epochs 45

# ── Data ablation: MAE with 25% labelled fine-tuning data ────────────────────
echo "=== MAE (25% labels): 30 pretrain + 45 finetune ==="
python3 audio_ast_mae.py \
  --output_dir data/runs/mae_pct25 \
  --pretrain_epochs 30 \
  --finetune_epochs 45 \
  --percent_train 25

# ── Data ablation: Scratch AST with 25% labelled data ────────────────────────
echo "=== Scratch AST (25% labels): 45 epochs supervised ==="
python3 audio_ast.py \
  --output_dir data/runs/scratch_pct25 \
  --epochs 45 \
  --percent_train 25

echo "=== All experiments complete ==="
echo "Run: python3 parse_logs.py && python3 make_charts.py && python3 make_training_curves.py"
