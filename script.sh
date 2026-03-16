#!/usr/bin/env bash
set -euo pipefail

echo "=== MAE smoke test: 1 pretrain epoch ==="
python3 audio_ast_mae.py \
  --output_dir data/audio_ast_mae_runs_smoke \
  --pretrain_epochs 1 \
  --finetune_epochs 0 > mae_smoke.log

echo "=== DINO smoke test: 1 pretrain epoch ==="
python3 audio_ast_dino.py \
  --output_dir data/audio_ast_dino_runs_smoke \
  --pretrain_epochs 1 \
  --finetune_epochs 0 > dino_smoke.log

echo "=== Pretrained AST smoke test: 1 epoch ==="
python3 audio_ast.py \
  --output_dir data/audio_ast_runs_smoke \
  --epochs 1 \
  --finetune > pretrain_smoke.log

echo "=== MAE full run: 30 pretrain epochs ==="
python3 audio_ast_mae.py \
  --output_dir data/audio_ast_mae_runs_full \
  --pretrain_epochs 30 > mae_30_15.log

echo "=== DINO full run: 30 pretrain epochs ==="
python3 audio_ast_dino.py \
  --output_dir data/audio_ast_dino_runs_full \
  --pretrain_epochs 30 > dino_30_15.log

echo "=== Pretrained AST full run: 15 epochs ==="
python3 audio_ast.py \
  --output_dir data/audio_ast_runs_full \
  --epochs 15 \
  --finetune > pretrained_15.log
