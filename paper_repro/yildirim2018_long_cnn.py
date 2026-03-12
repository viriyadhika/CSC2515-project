#!/usr/bin/env python3
"""
CNN baseline inspired by:

Yıldırım et al.
"Arrhythmia detection using deep convolutional neural network with long duration ECG signals"

Modified to match the benchmarking setup of the Tiny Transformer script:

Changes:
- 5-class AAMI setup (N/S/V/F/Q)
- HuggingFace Trainer
- identical metrics (accuracy, balanced_accuracy, macro_f1)
- beat-centered 10s fragments (3600 samples)
"""

from __future__ import annotations

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import wfdb

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

SEED = 42
FS = 360
SEG_LEN = 10 * FS
WINDOW = SEG_LEN // 2

EXCLUDED_RECORDS = {"102", "104", "107", "217"}

AAMI_MAP = {
    "N":0,"L":0,"R":0,"e":0,"j":0,
    "A":1,"a":1,"J":1,"S":1,
    "V":2,"E":2,
    "F":3,
    "/":4,"f":4,"Q":4,
}

IDX2CLS = {0:"N",1:"S",2:"V",3:"F",4:"Q"}

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zscore_per_segment(x, eps=1e-8):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < eps, eps, std)
    return (x - mean) / std


def encode_label(sym):
    return AAMI_MAP.get(sym)


def extract_fragments(folder):

    X, y = [], []

    records = sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".hea"))

    for rec in records:
        if rec in EXCLUDED_RECORDS:
            continue

        path = os.path.join(folder, rec)

        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")

        sig = record.p_signal[:,0]

        for peak, sym in zip(ann.sample, ann.symbol):

            lab = encode_label(sym)

            if lab is None:
                continue

            start = peak - WINDOW
            end = peak + WINDOW

            if start < 0 or end > len(sig):
                continue

            beat = sig[start:end]

            if len(beat) != SEG_LEN:
                continue

            X.append(beat)
            y.append(lab)

    return np.asarray(X), np.asarray(y)


class ECGDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return {
            "x": self.X[idx].unsqueeze(0),
            "labels": self.y[idx]
        }


class LongDurationCNN(nn.Module):

    def __init__(self, n_classes=5):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(1,32,7,padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32,64,5,padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64,128,5,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128,128,3,padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128,256,3,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256,256,3,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(16)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(256*16,512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,n_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x=None, labels=None):

        x = self.features(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss":loss,"logits":logits}


def data_collator(features):

    x = torch.stack([f["x"] for f in features])
    labels = torch.tensor([f["labels"] for f in features])

    return {"x":x,"labels":labels}


def compute_metrics(eval_pred):

    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    preds = np.argmax(logits,axis=1)

    return {
        "accuracy": accuracy_score(labels,preds),
        "balanced_accuracy": balanced_accuracy_score(labels,preds),
        "macro_f1": f1_score(labels,preds,average="macro",zero_division=0)
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder",type=str,default="mit-bih-arrhythmia-database-1.0.0")
    parser.add_argument("--epochs",type=int,default=40)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--lr",type=float,default=1e-3)

    args = parser.parse_args()

    seed_everything()

    X,y = extract_fragments(args.folder)

    X = zscore_per_segment(X)

    print("dataset size:",len(y))

    X_train,X_tmp,y_train,y_tmp = train_test_split(
        X,y,test_size=0.30,stratify=y,random_state=SEED
    )

    X_valid,X_test,y_valid,y_test = train_test_split(
        X_tmp,y_tmp,test_size=2/3,stratify=y_tmp,random_state=SEED
    )

    train_dataset = ECGDataset(X_train,y_train)
    valid_dataset = ECGDataset(X_valid,y_valid)
    test_dataset  = ECGDataset(X_test,y_test)

    model = LongDurationCNN()

    training_args = TrainingArguments(

        output_dir="./cnn_ecg",

        learning_rate=args.lr,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        num_train_epochs=args.epochs,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",

        greater_is_better=True,

        fp16=torch.cuda.is_available(),

        report_to="none",

        remove_unused_columns=False
    )

    trainer = Trainer(

        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=valid_dataset,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )

    trainer.train()

    print("Validation:",trainer.evaluate())
    print("Test:",trainer.evaluate(eval_dataset=test_dataset))

    pred = trainer.predict(test_dataset)

    y_pred = np.argmax(pred.predictions,axis=1)
    y_true = pred.label_ids

    print(classification_report(
        y_true,
        y_pred,
        labels=[0,1,2,3,4],
        target_names=[IDX2CLS[i] for i in range(5)],
        zero_division=0
    ))

    print(confusion_matrix(
        y_true,
        y_pred,
        labels=[0,1,2,3,4]
    ))


if __name__ == "__main__":
    main()