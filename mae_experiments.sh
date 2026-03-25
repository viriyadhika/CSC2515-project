# echo "=== MAE (AudioSet+ESC50): 45 pretrain + 45 finetune ==="
python3 audio_ast_mae.py \
  --output_dir data/runs/mae_big \
  --pretrain_epochs 45 \
  --finetune_epochs 45


echo "=== MAE (AudioSet+ESC50): 0 pretrain + 15 finetune ==="
python3 audio_ast_mae.py \
  --output_dir data/runs/mae_big_finetune_15_epochs \
  --pretrain_epochs 0 \
  --checkpoint data/runs/mae_big/pretrain/checkpoint-77220/model.safetensors \
  --finetune_epochs 15

# echo "=== DINO (AudioSet+ESC50): 45 pretrain + 45 finetune ==="
python3 audio_ast_dino.py \
  --output_dir data/runs/dino_big \
  --pretrain_epochs 45 \
  --finetune_epochs 45