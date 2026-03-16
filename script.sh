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

echo "=== Pretrained AST smoke test: 1 epoch ==="
python3 audio_ast.py \
  --output_dir data/audio_ast_runs_smoke \
  --epochs 1 \
  --finetune

echo "=== MAE full run: 30 pretrain epochs ==="
python3 audio_ast_mae.py \
  --output_dir data/audio_ast_mae_runs_full \
  --pretrain_epochs 30

echo "=== DINO full run: 30 pretrain epochs ==="
python3 audio_ast_dino.py \
  --output_dir data/audio_ast_dino_runs_full \
  --pretrain_epochs 30

echo "=== Pretrained AST full run: 30 epochs ==="
python3 audio_ast.py \
  --output_dir data/audio_ast_runs_full \
  --epochs 30 \
  --finetune
