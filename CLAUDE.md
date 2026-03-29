# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Empirical study of MAE and DINO self-supervised pretraining for audio event classification using Audio Spectrogram Transformers (AST) on ESC-50 (50-class, 2000 clips). JEPA code exists but was not used in the final report.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires CUDA GPU. Key deps: PyTorch 2.10, Transformers 5.3, Torchaudio 2.10.

---

## Entry-Point Scripts (`audio_ast_*.py`)

These are the three scripts that actually run experiments. Each follows the same structure: build transform → build datasets → pretrain (SSL only) → evaluate initial KNN → pretrain loop with milestone KNN checks → finetune on ESC-50 → write `metrics.json`.

### `audio_ast.py` — Supervised baselines
- Runs fine-tuning only (no SSL pretraining)
- `--finetune` flag loads pretrained HuggingFace SSAST weights; omitting it trains from random init
- Output dir layout: `trainer/trainer_state.json` (HF Trainer logs), `metrics.json`
- KNN milestones at epochs 15 and 30 measure fine-tuning progress (NOT pretraining — important distinction)

### `audio_ast_mae.py` — MAE pretraining + fine-tuning
Key flow:
1. Fit `pretrain_transform` (LogMelPadCrop) on AudioSet + ESC-50 waveforms
2. Fit `finetune_transform` separately on ESC-50 training waveforms only
3. Apply `percent_trained()` to subset training labels BEFORE building `embedding_train_dataset`
   - **Critical:** the initial KNN in a `--percent_train 25` run uses only 25% of training samples as the KNN reference gallery, so it will be lower than the epoch-45 KNN from the full pretraining run even if loading the same checkpoint
4. Load optional checkpoint via `--checkpoint`
5. Run `run_pretrain_loop()` (MAE objective, MSE loss on masked patches)
6. Run `run_finetune()` on ESC-50 with frozen backbone warmup
7. Write `metrics.json`

Key args:
- `--audioset_dir data/audioset_5000` — include AudioSet in pretraining corpus
- `--percent_train 25/50/100` — fraction of ESC-50 labels used for fine-tuning
- `--pretrain_epochs`, `--finetune_epochs`
- `--checkpoint path/to/model.safetensors` — resume from existing MAE checkpoint
- `--dataset_mean`, `--dataset_std` — override normalization constants (default: recompute from data)

### `audio_ast_dino.py` — DINO pretraining + fine-tuning
Same overall flow as MAE. Key differences:
- Augmentation applied inside `AudioDINOPretrainDataset`:
  - Default (soft): Gaussian noise (80%), time shift (50%), scale (50%), masking (30%)
  - Asymmetric (`--asymmetric`): teacher gets full spectrogram, student gets heavily time-masked view (~50% of frames in 3 contiguous blocks)
- `CollapseLoggingCallback` writes `pretrain/collapse_metrics.txt` with `epoch=N,mean_feature_std=X` per epoch — the primary diagnostic for representation collapse
- Gradient accumulation via `--gradient_accumulation_steps` (used in r2 experiments to get effective batch=64)

Key DINO hyperparameter args:
- `--out_dim` — projection head output dimension (256 or 1024)
- `--teacher_temp` — teacher softmax temperature (0.04 default, 0.07 tried)
- `--student_temp` — student softmax temperature (0.1)
- `--teacher_momentum` — EMA momentum (0.99)
- `--asymmetric` — use asymmetric augmentation

---

## Data Pipeline

### `LogMelPadCrop` (in `models/audio_common.py`)
Core transform: waveform → log-mel spectrogram → normalize → pad/crop to fixed size.

Parameters: sample_rate=16kHz, n_mels=128, n_fft=1024, hop_length=160, target_length=1024, top_db=80.
Output shape: (1024 time frames × 128 mel bins).

**Normalization:** `(spec - mean) / (std * 2.0)` where mean/std are computed by calling `.fit()` on a sample of training waveforms. The SSAST paper used fixed constants from 2M AudioSet clips (std ≈ 4 dB); ESC-50 actual std ≈ 12 dB — a 3× mismatch that severely hurt MAE training until corrected.

Two transforms are always fit separately:
- `pretrain_transform`: fit on AudioSet + ESC-50 combined
- `finetune_transform`: fit on ESC-50 training waveforms only

### Dataset classes (in `models/audio_common.py` and `novel/`)
- `WaveformArrayClassificationDataset` — ESC-50 labeled data for fine-tuning/KNN
- `WaveformArrayPretrainDataset` — ESC-50 waveforms for SSL pretraining (no labels)
- `FolderAudioPretrainDataset` — AudioSet audio files from a folder (no labels)
- `AudioDINOPretrainDataset` — wraps above with DINO augmentation

### ESC-50 data layout
- `data/esc50.csv` — metadata: filename, fold (1–5), target (class index), category (class name)
- `data/audio/audio/` — 2000 WAV files (5s each, 44.1kHz → resampled to 16kHz)
- Split: folds 1–3 = pretrain/train, fold 4 = validation, fold 5 = test
- `common/lib.py:load_esc50()` loads and splits the dataset

### AudioSet subset
- `data/audioset_5000/` — ~16K audio clips in subdirectories, used only for SSL pretraining
- Loaded by `FolderAudioPretrainDataset` which walks the directory recursively

---

## Model Architecture

### Backbone: SSAST
`Simon-Kotchou/ssast-small-patch-audioset-16-16` from HuggingFace.
- ViT-style, 12 layers, hidden size 384, 12 attention heads
- Input: (1024, 128) spectrogram → 512 patches of 16×16 each
- Loaded via `AutoModel.from_pretrained()`

### MAE (`models/mae_model.py`: `AudioASTMAE`)
- Masks ~250 of 512 patches randomly
- Encoder = SSAST backbone processes unmasked patches only
- Decoder = single TransformerEncoderLayer (hidden dim 256, num_heads computed via GCD)
- Reconstructs all 512 patches; loss = MSE on masked positions only
- `.clone_backbone()` returns just the encoder for KNN/fine-tuning

### DINO (`models/dino_model.py`: `AudioASTDINO`)
- Student and teacher share the same architecture (SSAST backbone + DINOHead)
- `DINOHead`: Linear(384→512) → GELU → Linear(512→128) → weight-normalized Linear(128→out_dim)
- Teacher = EMA of student, updated each step: `θ_t ← m·θ_t + (1-m)·θ_s`
- Center vector updated with momentum 0.9 to prevent collapse
- Loss = cross-entropy(student_softmax / τ_s, teacher_softmax / τ_t)

### Classifier (`models/audio_common.py`: `AudioASTClassifier`)
- Wraps backbone, mean-pools patch embeddings → Linear(384→50)
- During fine-tuning warmup (first 15 epochs of 45): backbone frozen, only head trained
- After warmup: backbone unfrozen with lr=1e-5, head lr=1e-4 (`_DifferentialLRTrainer`)

---

## Training Infrastructure

### `training/pretrain_loop.py`
- `run_pretrain_loop(model, train_dataset, eval_dataset, args, callbacks)` — wraps HuggingFace `Trainer`
- `run_finetune(backbone, finetune_data, args)` — supervised fine-tuning with differential LR
- `MilestoneEvalCallback` — triggers full KNN evaluation at specified epochs (15, 30, 45)
- `CollapseLoggingCallback` — computes mean feature std after each epoch, appends to `pretrain/collapse_metrics.txt`

### `common/metrics_logger.py`: `MetricsLogger`
- Accumulates results in memory, writes `metrics.json` at the end
- `set(key, value)` — top-level scalar (e.g. `initial_knn_acc`)
- `set_milestone(epoch, knn_acc, ...)` — nested under `milestones/{epoch}/`
- `set_final(val_acc, val_f1, test_acc, test_f1)` — final results

### `common/lib.py`
- `load_esc50(data_dir, csv_path)` → dict with X_train/X_val/X_test/y_train/... as numpy arrays of waveforms
- `percent_trained(X, y, args)` — subsets to `args.percent_train`% of training data (applied BEFORE building embedding dataset — affects KNN reference gallery size)
- `balance_classes()`, `compute_metrics()`, `set_seed()`

---

## Evaluation Pipeline

### `evaluation/embedding_eval.py`
- `compute_embeddings(backbone, dataset, batch_size)` — forward pass, mean-pool transformer output → (N, 384) numpy array
- `run_knn_eval(emb_train, y_train, emb_test, y_test)` — sklearn 5-NN, returns accuracy
- `evaluate_embedding_snapshots(backbone, train_dataset, test_dataset, ...)` — runs compute + KNN, saves results to `epoch_dir/knn_results.txt`

**Important:** KNN reference gallery = `embedding_train_dataset`, which is built from the (potentially subsetted) training split. At `percent_train=25`, only 25% of training samples are in the gallery → KNN accuracy is lower than the full-data pretraining milestone even with the same weights.

---

## Output Directory Structure

Each experiment writes to `data/runs/{experiment_name}/`:

```
{experiment_name}/
├── metrics.json                          # Final summary: initial_knn_acc, milestones, final
├── pretrain/
│   ├── checkpoint-{step}/
│   │   ├── model.safetensors             # Model weights
│   │   └── trainer_state.json            # HF Trainer log_history (loss per step)
│   └── collapse_metrics.txt              # DINO only: epoch=N,mean_feature_std=X
├── final_finetune/
│   └── checkpoint-{step}/
│       └── trainer_state.json            # Per-epoch train_loss, eval_loss, eval_accuracy
├── epoch_0/                              # KNN snapshot before any training
├── epoch_15/                             # KNN snapshot at pretrain epoch 15
└── epoch_30/                             # KNN snapshot at pretrain epoch 30
```

Supervised baselines (`pretrained`, `scratch`) use `trainer/trainer_state.json` instead of `final_finetune/`.

### `metrics.json` schema
```json
{
  "initial_knn_acc": 0.1175,
  "milestones": {
    "15": { "knn_acc": 0.215 },
    "30": { "knn_acc": 0.2975 },
    "45": { "knn_acc": 0.325 }
  },
  "final": {
    "val_acc": 0.5375, "val_f1": 0.506,
    "test_acc": 0.4975, "test_f1": 0.482
  }
}
```

---

## All Experiment Runs (`data/runs/`)

### Supervised baselines
| Folder | Notes | Test Acc |
|--------|-------|----------|
| `pretrained` | HF SSAST weights fine-tuned | 73.0% |
| `scratch` | Random init fine-tuned | 40.0% |

### MAE experiments
| Folder | Notes | Test Acc |
|--------|-------|----------|
| `mae_big_preprocess` | **Best SSL.** AudioSet+ESC50, correct normalization | 49.75% |
| `mae_esc50_preprocess` | **Used in report.** ESC50-only, correct normalization | 27.5% |
| `mae_big` | AudioSet+ESC50, wrong normalization (paper default) | 31.25% |
| `mae_esc50_only` | ESC50-only, wrong normalization — **not used in report** | 34.25% |
| `mae` | Earlier intermediate experiment | 36.0% |

### MAE label-efficiency ablations (all start from `mae_big_preprocess` checkpoint)
| Folder | Labels | Test Acc |
|--------|--------|----------|
| `mae_50_pct` | 50% | 39.5% |
| `mae_25_pct` | 25% | 27.5% |
| `mae_pct25` | 25%, old pretraining (wrong norm) | 16.0% — **stale, replaced by mae_25_pct** |

### Scratch label-efficiency ablations
| Folder | Labels | Test Acc |
|--------|--------|----------|
| `scratch_50_pct` | 50% | 26.25% |
| `scratch_pct25` | 25% | 21.0% |

### DINO Round 1 (batch size 8)
| Folder | Config | Test Acc |
|--------|--------|----------|
| `dino` | Baseline, wrong norm | 23.25% |
| `dino_big_preprocess` | Soft aug, out=256, correct norm | 21.0% |
| `dino_exp1_outdim1024` | Soft aug, out=1024 | 17.0% |
| `dino_exp2_temp007` | Soft aug, out=1024, τ_t=0.07 | 16.5% |
| `dino_exp3_asymmetric` | Asymmetric aug — **collapsed**, stopped at ep15 | — |

### DINO Round 2 (batch size 64 via gradient accumulation ×8)
| Folder | Config | Test Acc |
|--------|--------|----------|
| `dino_r2_exp1_accum` | Soft aug, out=256 | 14.25% |
| `dino_r2_exp2_accum_1024_t07` | Soft aug, out=1024, τ_t=0.07 | 18.75% |
| `dino_r2_exp3_accum_asymmetric` | Asymmetric aug, out=1024 — **no collapse** | 21.0% |

### `data/runs/summary/`
Generated by the analysis scripts. Contains CSVs and chart PNGs.

---

## Data Analysis Scripts

### `parse_logs.py`
- Reads `metrics.json` from each experiment folder listed in `EXPERIMENTS`
- Outputs two CSVs to `data/runs/summary/`:
  - `audio_ast_results.csv` — full table with all milestone columns
  - `audio_ast_key_results.csv` — compact key columns only
- **Must be run first** before `make_charts.py` (charts read from the CSV)
- To add a new experiment: append an entry to `EXPERIMENTS` list with `output_dir`, `method`, `percent_train`, etc.

### `make_charts.py`
Reads `data/runs/summary/audio_ast_results.csv` and generates all comparison charts.

| Chart | File | Content |
|-------|------|---------|
| 1 | `chart1_model_comparison.png` | Bar chart: final test accuracy, all main methods |
| 3 | `chart3_knn_accuracy.png` | SSL methods: 4 KNN bars (ep 0/15/30/45); Pretrained/Scratch: initial only |
| 4 | `chart4_data_ablation.png` | Line chart: test acc vs label fraction (25/50/100%) for MAE vs Scratch |
| 5 | `chart5_mae_data_source.png` | Two panels: final acc + KNN line chart for AudioSet vs ESC50-only |
| 7 | `chart7_mae_preprocessing.png` | mae_big vs mae_big_preprocess: KNN progression + final acc |
| 8 | `chart8_dino_collapse.png` | mean_feature_std over pretraining epochs (clipped at 30) for all DINO variants |

Charts 2, 6a, 6b are generated but **not used in the report** — don't add them back.

Also reads `data/runs/*/pretrain/collapse_metrics.txt` directly (chart8) — not via CSV.

### `make_training_curves.py`
Reads `trainer_state.json` files directly (bypasses CSV). Hardcoded paths in `RUNS` dict:
- `MAE` → `mae_big_preprocess/final_finetune/checkpoint-6750/trainer_state.json`
- `DINO` → `dino/final_finetune/checkpoint-6750/trainer_state.json`
- `MAE ESC-50 Only` → `mae_esc50_preprocess/final_finetune/checkpoint-6750/trainer_state.json`
- `Pretrained` → `pretrained/trainer/trainer_state.json`
- `Scratch` → `scratch/trainer/trainer_state.json`

`EXCLUDE_FROM_MAIN = {"Pretrained"}` — Pretrained is excluded from left panel of combined chart.

Outputs (all saved to `data/runs/summary/`):
- `chart6_val_acc_combined.png` — **used in report**: two-panel val accuracy (SSL+Scratch left, MAE vs Pretrained right)
- `chart6b_loss_curves.png` — **used in report**: train loss + val loss for MAE, MAE ESC-50 Only, Scratch
- `chart6_finetune_curves.png` — not used in report (3-panel with all models)

---

## Report

- **Location:** `data/report/report.tex` + all PNGs in the same folder
- **Style:** NeurIPS 2025 (`neurips_2025.sty` — extracted from the zip downloaded from NeurIPS CDN)
- **Compile:** always run pdflatex **twice** from `data/report/` — single pass leaves `??` for cross-references
  ```bash
  cd data/report && pdflatex report.tex && pdflatex report.tex
  ```
- The `! Undefined control sequence` error on every compile is from `\@trackname` in the NeurIPS sty file itself — harmless, PDF is produced
- **Figures in report** (8 PNGs, all must be present in `data/report/`):
  `chart1`, `chart3`, `chart4`, `chart5`, `chart6_val_acc_combined`, `chart6b_loss_curves`, `chart7`, `chart8`

### Typical workflow to update report after new experiment data:
```bash
cd /path/to/project
python3 parse_logs.py          # regenerate CSVs
python3 make_charts.py         # regenerate charts 1,3,4,5,7,8
python3 make_training_curves.py # regenerate chart6_val_acc_combined, chart6b_loss_curves
cp data/runs/summary/chart*.png data/report/
cd data/report && pdflatex report.tex && pdflatex report.tex
```

---

## Files to Ignore

These exist but are **not part of the final pipeline**:
- `audio_ast_jepa.py`, `jepa.py` — JEPA experiments, not completed
- `mae.py`, `dino.py` — ECG/early experiments on MIT-BIH arrhythmia data
- `mae_freq.py`, `tiny_transformer_2402.py`, `faildetection.py` — exploratory, not used
- `paper_repro/` — CNN/BiLSTM reproductions from prior papers
- `audio/` — `bc_utils.py`, `utils.py`, `utils2.py` — audio utility code not used in main pipeline
- `download_audioset.py` — script used to download the AudioSet subset (already downloaded)
- `data/runs/mae_pct25/` — stale experiment replaced by `mae_25_pct`
- `data/runs/mae/` — intermediate early experiment, not reported
