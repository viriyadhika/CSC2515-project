#!/usr/bin/env bash
set -euo pipefail

echo "=== MAE smoke test: 1 pretrain epoch ==="
python3 audio_ast_mae.py \
  --output_dir data/audio_ast_mae_runs_smoke \
  --pretrain_epochs 1 \
  --finetune_epochs 0

echo "=== DINO smoke test: 1 pretrain epoch ==="
python3 audio_ast_dino.py \
  --output_dir data/audio_ast_dino_runs_smoke \
  --pretrain_epochs 1 \
  --finetune_epochs 0

echo "=== MAE full run: 40 pretrain epochs ==="
python3 audio_ast_mae.py \
  --output_dir data/audio_ast_mae_runs_full \
  --pretrain_epochs 40

echo "=== DINO full run: 40 pretrain epochs ==="
python3 audio_ast_dino.py \
  --output_dir data/audio_ast_dino_runs_full \
  --pretrain_epochs 40
