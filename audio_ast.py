#!/usr/bin/env python3
"""
Hugging Face AST finetuning on ESC-50 with milestone logging.

This uses a pretrained AST checkpoint and logs:
- KNN accuracy
- t-SNE / PCA grouped plots
- validation/test/classification metrics

at epochs 15 and 30 while training continues.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
)

from common.dataloader import AudioLoader
from common.lib import SEED, compute_metrics, make_training_args, percent_trained, seed_everything
from evaluation.embedding_eval import evaluate_embedding_snapshots
from training.pretrain_loop import EVAL_EPOCHS


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
        x: np.ndarray,
        y: np.ndarray,
        feature_extractor,
        source_sampling_rate: int,
    ):
        self.labels = torch.tensor(y, dtype=torch.long)
        target_sampling_rate = int(feature_extractor.sampling_rate)

        features = []
        for waveform in x:
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


def write_metrics_snapshot(trainer, test_dataset, label_names, n_classes, output_dir):
    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)

    pred_output = trainer.predict(test_dataset)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids
    label_ids = list(range(n_classes))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)

    output_path = Path(output_dir) / "finetune_metrics.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "Validation metrics:",
                str(val_metrics),
                "",
                "Test metrics:",
                str(test_metrics),
                "",
                "Classification report:",
                report,
                "",
                "Confusion matrix:",
                np.array2string(matrix),
            ]
        )
        + "\n"
    )

    return val_metrics, test_metrics, report, matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=AST_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./audio_ast_runs")
    parser.add_argument("--percent_train", type=float, default=100.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)

    dataset = AudioLoader(args).load()
    x_train = dataset["X_train"]
    x_valid = dataset["X_valid"]
    x_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_valid = dataset["y_valid"]
    y_test = dataset["y_test"]
    label_names = dataset["label_names"]
    n_classes = int(dataset["n_classes"])

    x_train, y_train = percent_trained(x_train, y_train, args)

    feature_extractor_name = args.checkpoint or args.model_name
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)

    train_dataset = ASTAudioDataset(
        x_train,
        y_train,
        feature_extractor=feature_extractor,
        source_sampling_rate=8000,
    )
    valid_dataset = ASTAudioDataset(
        x_valid,
        y_valid,
        feature_extractor=feature_extractor,
        source_sampling_rate=8000,
    )
    test_dataset = ASTAudioDataset(
        x_test,
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

    for epoch in range(1, args.epochs + 1):
        training_args = make_training_args(
            output_dir=str(Path(args.output_dir) / f"epoch_{epoch}" / "trainer"),
            epochs=1,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=SEED,
        )
        training_args.metric_for_best_model = "macro_f1"
        training_args.greater_is_better = True
        training_args.load_best_model_at_end = False
        training_args.save_strategy = "no"

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=ast_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        model = trainer.model
        print(f"Finished supervised epoch {epoch}")

        if epoch in EVAL_EPOCHS:
            epoch_dir = Path(args.output_dir) / f"epoch_{epoch}"
            backbone = model.audio_spectrogram_transformer
            snapshot = evaluate_embedding_snapshots(
                backbone=backbone,
                train_dataset=train_dataset,
                y_train=y_train,
                test_dataset=test_dataset,
                y_test=y_test,
                output_dir=epoch_dir,
                idx2cls={i: label_names[i] for i in range(n_classes)},
                batch_size=args.batch_size,
                input_key="input_values",
            )
            print(f"Epoch {epoch} KNN accuracy: {snapshot['knn_accuracy']:.4f}")

            val_metrics, test_metrics, _, _ = write_metrics_snapshot(
                trainer=trainer,
                test_dataset=test_dataset,
                label_names=label_names,
                n_classes=n_classes,
                output_dir=epoch_dir,
            )
            print(f"Epoch {epoch} validation metrics: {val_metrics}")
            print(f"Epoch {epoch} test metrics: {test_metrics}")

    final_dir = Path(args.output_dir) / "final"
    val_metrics, test_metrics, report, matrix = write_metrics_snapshot(
        trainer=trainer,
        test_dataset=test_dataset,
        label_names=label_names,
        n_classes=n_classes,
        output_dir=final_dir,
    )

    print("Validation metrics:")
    print(val_metrics)
    print("Test metrics:")
    print(test_metrics)
    print(report)
    print(matrix)


if __name__ == "__main__":
    main()
