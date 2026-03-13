#!/usr/bin/env python3
"""
Hugging Face AST finetuning on ESC-50.

This follows the training structure already used in `mae.py` / `mae_freq.py`:
- shared dataset loading from `common.dataloader.AudioLoader`
- Hugging Face `Trainer` training/evaluation
- final classification report + confusion matrix on the held-out test fold

The implementation uses a pretrained AST checkpoint from Hugging Face and
finetunes it on ESC-50.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, AutoConfig

from common.dataloader import AudioLoader
from common.lib import (
    SEED,
    seed_everything,
    compute_metrics,
    make_training_args,
    percent_trained,
)


AST_MODEL_NAME = "Simon-Kotchou/ssast-small-patch-audioset-16-16"


def resample_waveform(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if source_rate == target_rate:
        return waveform

    new_length = int(round(len(waveform) * target_rate / source_rate))
    x = torch.tensor(waveform, dtype=torch.float32).view(1, 1, -1)
    y = F.interpolate(x, size=new_length, mode="linear", align_corners=False)
    return y.view(-1).cpu().numpy()


class ASTAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_extractor,
        source_sampling_rate: int,
    ):
        self.labels = torch.tensor(y, dtype=torch.long)
        target_sampling_rate = int(feature_extractor.sampling_rate)

        features = []
        for waveform in X:
            waveform = resample_waveform(
                waveform,
                source_rate=source_sampling_rate,
                target_rate=target_sampling_rate,
            )
            encoded = feature_extractor(
                waveform,
                sampling_rate=target_sampling_rate,
                return_tensors="pt",
            )
            features.append(encoded["input_values"][0])

        self.input_values = torch.stack(features)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_values": self.input_values[idx],
            "labels": self.labels[idx],
        }


def ast_collator(features):
    input_values = torch.stack([f["input_values"] for f in features])
    labels = torch.tensor([int(f["labels"]) for f in features], dtype=torch.long)
    return {"input_values": input_values, "labels": labels}


def ast_finetune_from_datasets(
    model,
    train_dataset,
    valid_dataset,
    test_dataset,
    training_args,
):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=ast_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    return trainer, val_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=AST_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./audio_ast_runs")
    parser.add_argument(
        "--percent_train",
        type=float,
        default=100.0,
        help="Percent of labeled training data to use for finetuning (0-100)",
    )
    parser.add_argument("--balance_target_size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)

    audio_loader = AudioLoader(args)
    dataset = audio_loader.load()
    X_train = dataset["X_train"]
    X_valid = dataset["X_valid"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_valid = dataset["y_valid"]
    y_test = dataset["y_test"]
    label_names = dataset["label_names"]
    n_classes = int(dataset["n_classes"])

    X_train, y_train = percent_trained(X_train, y_train, args)

    feature_extractor_name = args.checkpoint or args.model_name
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)

    train_dataset = ASTAudioDataset(
        X_train,
        y_train,
        feature_extractor=feature_extractor,
        source_sampling_rate=8000,
    )
    valid_dataset = ASTAudioDataset(
        X_valid,
        y_valid,
        feature_extractor=feature_extractor,
        source_sampling_rate=8000,
    )
    test_dataset = ASTAudioDataset(
        X_test,
        y_test,
        feature_extractor=feature_extractor,
        source_sampling_rate=8000,
    )

    id2label = {i: label_names[i] for i in range(n_classes)}
    label2id = {label_names[i]: i for i in range(n_classes)}

    model_name = args.checkpoint or args.model_name

    if args.finetune:
        print("Loading pretrained model")
        model = AutoModelForAudioClassification.from_pretrained(
            model_name,
            num_labels=n_classes,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
    else:
        print("Loading raw model")
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=n_classes,
            label2id=label2id,
            id2label=id2label,
        )

        model = AutoModelForAudioClassification.from_config(config)

    training_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "finetune"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=SEED,
    )
    training_args.metric_for_best_model = "macro_f1"
    training_args.greater_is_better = True

    trainer, val_metrics, test_metrics = ast_finetune_from_datasets(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        training_args=training_args,
    )

    print("Validation metrics:")
    print(val_metrics)

    print("Test metrics:")
    print(test_metrics)

    pred_output = trainer.predict(test_dataset)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids

    label_ids = list(range(n_classes))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=label_names,
            zero_division=0,
        )
    )
    print(confusion_matrix(y_true, y_pred, labels=label_ids))


if __name__ == "__main__":
    main()
