import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm

from common.lib import (
    compute_metrics,
    seed_everything,
    SEED,
    make_training_args,
    extract_beats_and_rr,
    preprocess_beats,
    balance_classes,
    percent_trained,
)

window = 93
folder = "data/mit-bih-arrhythmia-database-1.0.0"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder",
    type=str,
    default=folder,
    help="Path to MIT-BIH records (same as other scripts).",
)
parser.add_argument(
    "--window",
    type=int,
    default=window,
    help="Half-window size around R peak (beat length = 2*window+1).",
)
parser.add_argument(
    "--percent_train",
    type=float,
    default=100.0,
    help="Percent of labeled training data to use for training (0-100).",
)
parser.add_argument(
    "--balance_target",
    type=int,
    default=5000,
    help="Per-class target size for training rebalancing.",
)
args = parser.parse_args()

window = args.window
folder = args.folder

X, RR, y = extract_beats_and_rr(
    folder,
    pre_process=None,
    window=window,
)

print(X.shape, y.shape)

X = preprocess_beats(X)
print("After shared preprocessing:", X.shape)

# 7:1:2 stratified split, matching mae/dino/mae_freq
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=SEED
)

# Subsample labeled training data if requested
X_train, y_train = percent_trained(X_train, y_train, args)

# Rebalance training set only; valid/test keep real distribution
X_train, y_train = balance_classes(
    X_train,
    y_train,
    target_size=args.balance_target,
    seed=SEED,
    n_classes=5,
)

print("Train set after balancing:", X_train.shape)
print("Valid set:", X_valid.shape)
print("Test set:", X_test.shape)

# Convert to tensors and add channel dim [B, 1, L]
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)

X_valid = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(1)
y_valid = torch.tensor(y_valid, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.long)

# %%
class ECGDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return {
            "input": self.X[idx],
            "labels": self.y[idx]
        }

train_dataset = ECGDataset(X_train, y_train)
valid_dataset = ECGDataset(X_valid, y_valid)
test_dataset = ECGDataset(X_test, y_test)

# %%
import torch.nn.functional as F
import torch.nn as nn

class ECGCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1,32,5)
        self.conv2 = nn.Conv1d(32,64,5)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(2752, 128)
        self.fc2 = nn.Linear(128,5)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, labels=None):

        x = self.pool(F.relu(self.conv1(input)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECGCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# %%
training_args = make_training_args(
    output_dir="./ecg_model",
    epochs=15,
    batch_size=128,
    lr=1e-3,
    seed=SEED,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()
print("Validation metrics:")
val_metrics = trainer.evaluate()
print(val_metrics)

print("Test metrics:")
test_metrics = trainer.evaluate(eval_dataset=test_dataset)
print(test_metrics)

predictions = trainer.predict(test_dataset)

logits = predictions.predictions
labels = predictions.label_ids

preds = logits.argmax(axis=1)

print(classification_report(labels, preds))

print(confusion_matrix(labels, preds))

# %%



