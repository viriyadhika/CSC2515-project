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

echo "=== Scratch AST smoke test: 1 epoch ==="
python3 audio_ast.py \
  --output_dir data/audio_ast_scratch_runs_smoke \
  --epochs 1 > scratch_smoke.log

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

echo "=== Scratch AST full run: 15 epochs ==="
python3 audio_ast.py \
  --output_dir data/audio_ast_scratch_runs_full \
  --epochs 15 > scratch_15.log

echo "=== Scratch AST long run: 45 epochs ==="
python3 audio_ast.py \
  --output_dir data/audio_ast_scratch_runs_45 \
  --epochs 45 > scratch_45.log

echo "=== MAE checkpoint finetune: 45 epochs from best checkpoint ==="
python3 audio_ast_mae.py \
  --output_dir data/audio_ast_mae_ckpt_finetune_45 \
  --checkpoint data/audio_ast_mae_runs_full/pretrain/best.pt \
  --pretrain_epochs 0 \
  --finetune_epochs 45 \
  --run_final_finetune > mae_ckpt_finetune_45.log

echo "=== MAE 25 percent data ablation ==="
python3 audio_ast_mae.py \
  --output_dir data/audio_ast_mae_pct25 \
  --checkpoint data/audio_ast_mae_runs_full/pretrain/best.pt \
  --pretrain_epochs 0 \
  --finetune_epochs 15 \
  --percent_train 25 \
  --run_final_finetune > mae_pct25.log

echo "=== Scratch AST 25 percent data ablation ==="
python3 audio_ast.py \
  --output_dir data/audio_ast_scratch_pct25 \
  --epochs 15 \
  --percent_train 25 > scratch_pct25.log

echo "=== MAE ESC50-only pretraining ==="
python3 audio_ast_mae.py \
  --output_dir data/audio_ast_mae_esc50_only \
  --pretrain_epochs 30 \
  --esc50_only_pretrain > mae_esc50_only.log
