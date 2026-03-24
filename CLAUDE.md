# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project comparing self-supervised learning (SSL) methods (MAE, DINO, JEPA) for audio event classification using Audio Spectrogram Transformers (AST) on the ESC-50 dataset, for UofT CSC2515.

## Setup

```bash
pip install -r requirements.txt
```

Requires CUDA-capable GPU. Key deps: PyTorch 2.10, Transformers 5.3, Torchaudio 2.10.

## Running Experiments

```bash
# Supervised baselines (pretrained AST or from scratch)
python3 audio_ast.py --output_dir data/runs/pretrained --epochs 15 --finetune
python3 audio_ast.py --output_dir data/runs/scratch --epochs 15

# SSL pretraining + ESC-50 finetuning
python3 audio_ast_mae.py --output_dir data/runs/mae --pretrain_epochs 30 --finetune_epochs 15
python3 audio_ast_dino.py --output_dir data/runs/dino --pretrain_epochs 30 --finetune_epochs 15
python3 audio_ast_jepa.py --output_dir data/runs/jepa --pretrain_epochs 30 --finetune_epochs 15

# Full experiment suite
bash script.sh
```

Common optional args: `--audioset_dir data/audioset_5000`, `--percent_train 0.5` (data ablation), `--batch_size`, `--learning_rate`.

## Architecture

**Backbone:** `Simon-Kotchou/ssast-small-patch-audioset-16-16` (HuggingFace AST, 16×16 patches on mel-spectrogram)

**Data pipeline:** Raw waveform → `LogMelPadCrop` (mel-spectrogram, normalized) → patch tokens → SSL pretraining → frozen backbone → classification head

**SSL methods:**
- **MAE** (`models/mae_model.py`): Masks ~100 patches, reconstructs them with a 1-layer transformer decoder. Loss: MSE in patch space.
- **DINO** (`models/dino_model.py`): Student-teacher with EMA momentum updates. Student and teacher share architecture with MLP projection heads. Loss: cross-entropy between student/teacher softmax outputs.
- **JEPA** (`audio_ast_jepa.py`): Predicts target patch embeddings from context patches (60% mask ratio). Loss: MSE in embedding space.

**Training infrastructure** (`training/pretrain_loop.py`): Uses HuggingFace `Trainer` with `MilestoneEvalCallback` that runs full evaluation at epochs 15 and 30. `CollapseLoggingCallback` monitors feature std to detect representation collapse.

**Evaluation pipeline** (`evaluation/embedding_eval.py`): At each milestone — extract frozen embeddings, run 5-NN KNN accuracy, generate t-SNE/PCA plots, run temporary finetuning for downstream accuracy.

## Data

- **ESC-50:** `data/esc50.csv` (metadata) + `data/audio/audio/` (audio clips). 50 classes, 5 folds; folds 1–3 for pretraining/training, fold 4 for val, fold 5 for test.
- **AudioSet subset:** `data/audioset_5000/` (optional, used for SSL pretraining only)
- **ECG data:** `data/mit-bih-arrhythmia-database-1.0.0/` (for ECG experiments: `mae.py`, `dino.py`, `jepa.py`)

## Key Modules

| File | Role |
|------|------|
| `models/audio_common.py` | `LogMelPadCrop`, `AudioASTClassifier`, dataset wrappers |
| `models/mae_model.py` | `AudioASTMAE` |
| `models/dino_model.py` | `AudioASTDINO` |
| `novel/dino_utils.py` | DINO collator, callbacks, training helpers |
| `novel/mae_lib.py` | MAE collator, shared CLI arg definitions |
| `common/lib.py` | Seed utilities, metrics, dataset helpers |
| `training/pretrain_loop.py` | `run_pretrain_loop()`, `run_finetune()`, callbacks |
| `evaluation/embedding_eval.py` | `compute_embeddings()`, `run_knn_eval()`, `evaluate_embedding_snapshots()` |
