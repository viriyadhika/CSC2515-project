# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pywt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from common.lib import (
    compute_metrics,
    seed_everything,
    SEED,
    make_training_args,
    extract_beats_and_rr,
    preprocess_beats_and_balance,
)

# %%

# %%
import numpy as np
from sklearn.model_selection import train_test_split

window = 93

# %%

folder = "data/mit-bih-arrhythmia-database-1.0.0"

X, RR, y = extract_beats_and_rr(
    folder,
    pre_process=None,
    window=window,
)

print(X.shape, y.shape)

X, y = preprocess_beats_and_balance(
    X,
    y,
    target_size=5000,
    seed=SEED,
    n_classes=5,
)

print("Balanced dataset:", X.shape)

# %%
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# add channel dimension for Conv1D
X = X.unsqueeze(1)

print(X.shape)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

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

# %%
train_dataset = ECGDataset(X_train, y_train)
test_dataset = ECGDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128
)

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
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# %%
trainer.train()
metrics = trainer.evaluate()
print(metrics)


predictions = trainer.predict(test_dataset)

logits = predictions.predictions
labels = predictions.label_ids

preds = logits.argmax(axis=1)

print(classification_report(labels, preds))

print(confusion_matrix(labels, preds))

# %%



