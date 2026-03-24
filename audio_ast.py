#!/usr/bin/env python3
"""
Supervised AST training on ESC-50 (pre-trained or from scratch).

Runs a single HuggingFace Trainer for --epochs epochs. KNN and val/test
accuracy are logged to {output_dir}/metrics.json at epochs 15 and 30.
Per-epoch training curves are in {output_dir}/trainer/trainer_state.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score as sklearn_f1_score
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainerCallback,
)

from common.dataloader import AudioLoader
from common.lib import SEED, compute_metrics, make_training_args, percent_trained, seed_everything
from common.metrics_logger import MetricsLogger
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
    def __init__(self, x, y, feature_extractor, source_sampling_rate: int):
        self.labels = torch.tensor(y, dtype=torch.long)
        target_rate = int(feature_extractor.sampling_rate)
        features = []
        for waveform in x:
            waveform = resample_waveform(waveform, source_rate=source_sampling_rate,
                                         target_rate=target_rate)
            encoded = feature_extractor(waveform, sampling_rate=target_rate,
                                        return_tensors="pt")
            features.append(encoded["input_values"][0])
        self.input_values = torch.stack(features)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {"input_values": self.input_values[idx], "labels": self.labels[idx]}


def ast_collator(features):
    return {
        "input_values": torch.stack([f["input_values"] for f in features]),
        "labels": torch.tensor([int(f["labels"]) for f in features], dtype=torch.long),
    }


class SupervisedMilestoneCallback(TrainerCallback):
    """Logs KNN + val/test accuracy to metrics.json at milestone epochs."""

    def __init__(
        self,
        train_dataset,
        y_train,
        test_dataset,
        y_test,
        idx2cls,
        batch_size,
        metrics_logger: MetricsLogger,
        output_dir,
    ):
        self.train_dataset = train_dataset
        self.y_train = y_train
        self.test_dataset = test_dataset
        self.y_test = y_test
        self.idx2cls = idx2cls
        self.batch_size = batch_size
        self.metrics_logger = metrics_logger
        self.output_dir = Path(output_dir)
        self.trainer = None  # set after Trainer creation

    def _infer(self, model, dataset):
        model.eval()
        device = next(model.parameters()).device
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=ast_collator
        )
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                all_preds.extend(out.logits.argmax(-1).cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        return np.array(all_preds), np.array(all_labels)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        if epoch not in EVAL_EPOCHS:
            return

        # Val accuracy from trainer state (already evaluated via eval_strategy="epoch")
        val_acc = val_f1 = None
        for log in reversed(state.log_history):
            if "eval_accuracy" in log and abs(log.get("epoch", 0) - epoch) < 0.5:
                val_acc = log.get("eval_accuracy")
                val_f1 = log.get("eval_macro_f1")
                break

        # Test metrics via direct inference
        test_preds, test_labels = self._infer(model, self.test_dataset)
        test_acc = float((test_preds == test_labels).mean())
        test_f1 = float(sklearn_f1_score(test_labels, test_preds, average="macro", zero_division=0))

        # KNN on frozen backbone embeddings
        backbone = model.audio_spectrogram_transformer
        snapshot = evaluate_embedding_snapshots(
            backbone=backbone,
            train_dataset=self.train_dataset,
            y_train=self.y_train,
            test_dataset=self.test_dataset,
            y_test=self.y_test,
            output_dir=self.output_dir / f"epoch_{epoch}",
            idx2cls=self.idx2cls,
            batch_size=self.batch_size,
            input_key="input_values",
            skip_viz=True,
        )

        self.metrics_logger.set_milestone(
            epoch=epoch,
            knn_acc=snapshot["knn_accuracy"],
            val_acc=val_acc,
            val_f1=val_f1,
            test_acc=test_acc,
            test_f1=test_f1,
        )
        print(
            f"Epoch {epoch} | KNN: {snapshot['knn_accuracy']:.4f} | "
            f"val_acc: {val_acc} | test_acc: {test_acc:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=AST_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="data/runs/pretrained")
    parser.add_argument("--percent_train", type=float, default=100.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)
    metrics_logger = MetricsLogger(args.output_dir)

    dataset = AudioLoader(args).load()
    x_train, y_train = percent_trained(dataset["X_train"], dataset["y_train"], args)
    x_valid = dataset["X_valid"]
    x_test = dataset["X_test"]
    y_valid = dataset["y_valid"]
    y_test = dataset["y_test"]
    label_names = dataset["label_names"]
    n_classes = int(dataset["n_classes"])

    feature_extractor_name = args.checkpoint or args.model_name
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)

    train_dataset = ASTAudioDataset(x_train, y_train, feature_extractor, source_sampling_rate=8000)
    valid_dataset = ASTAudioDataset(x_valid, y_valid, feature_extractor, source_sampling_rate=8000)
    test_dataset = ASTAudioDataset(x_test, y_test, feature_extractor, source_sampling_rate=8000)

    id2label = {i: label_names[i] for i in range(n_classes)}
    label2id = {label_names[i]: i for i in range(n_classes)}
    model_name = args.checkpoint or args.model_name

    if args.finetune:
        print("Loading pre-trained model weights.")
        model = AutoModelForAudioClassification.from_pretrained(
            model_name, num_labels=n_classes, label2id=label2id, id2label=id2label,
            ignore_mismatched_sizes=True,
        )
    else:
        print("Initialising model from random weights (scratch).")
        config = AutoConfig.from_pretrained(
            model_name, num_labels=n_classes, label2id=label2id, id2label=id2label,
        )
        model = AutoModelForAudioClassification.from_config(config)

    idx2cls = {i: label_names[i] for i in range(n_classes)}

    # Initial KNN before any training
    initial_snapshot = evaluate_embedding_snapshots(
        backbone=model.audio_spectrogram_transformer,
        train_dataset=train_dataset,
        y_train=y_train,
        test_dataset=test_dataset,
        y_test=y_test,
        output_dir=Path(args.output_dir) / "epoch_0",
        idx2cls=idx2cls,
        batch_size=args.batch_size,
        input_key="input_values",
        skip_viz=True,
    )
    metrics_logger.set("initial_knn_acc", initial_snapshot["knn_accuracy"])
    print(f"Initial KNN accuracy: {initial_snapshot['knn_accuracy']:.4f}")

    milestone_cb = SupervisedMilestoneCallback(
        train_dataset=train_dataset,
        y_train=y_train,
        test_dataset=test_dataset,
        y_test=y_test,
        idx2cls=idx2cls,
        batch_size=args.batch_size,
        metrics_logger=metrics_logger,
        output_dir=args.output_dir,
    )

    training_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "trainer"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=SEED,
    )
    # For supervised runs we don't need to checkpoint every epoch
    training_args.save_strategy = "no"
    training_args.load_best_model_at_end = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=ast_collator,
        compute_metrics=compute_metrics,
        callbacks=[milestone_cb],
    )
    milestone_cb.trainer = trainer  # needed for potential future use
    trainer.train()

    # Final metrics
    val_m = trainer.evaluate()
    test_preds, test_labels = milestone_cb._infer(model, test_dataset)
    test_acc = float((test_preds == test_labels).mean())
    test_f1 = float(sklearn_f1_score(test_labels, test_preds, average="macro", zero_division=0))

    metrics_logger.set_final(
        val_acc=val_m.get("eval_accuracy"),
        val_f1=val_m.get("eval_macro_f1"),
        test_acc=test_acc,
        test_f1=test_f1,
    )
    print(f"Final val accuracy:  {val_m.get('eval_accuracy')}")
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
