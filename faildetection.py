# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pywt
from scipy import stats

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# %%
def denoise(signal):

    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(signal), w.dec_len)

    threshold = 0.04

    coeffs = pywt.wavedec(signal, 'sym4', level=maxlev)

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    reconstructed = pywt.waverec(coeffs, 'sym4')

    return reconstructed[:len(signal)]

# %%
import wfdb
import numpy as np
from sklearn.model_selection import train_test_split

window = 93


def encode_label(l):

    if l in ['N','L','R','e','j']:
        return 0

    elif l in ['A','a','J','S']:
        return 1

    elif l in ['V','E']:
        return 2

    elif l in ['F']:
        return 3

    elif l in ['/', 'f', 'Q']:
        return 4

    return None


EXCLUDED_RECORDS = {"102", "104", "107", "217"}


def normalize(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)


def extract_beats(record_list, folder):

    beats = []
    labels = []

    for r in record_list:

        if r in EXCLUDED_RECORDS:
            continue

        path = f"{folder}/{r}"

        record = wfdb.rdrecord(path)
        annotation = wfdb.rdann(path, 'atr')

        signal = record.p_signal[:,0]
        peaks = annotation.sample
        ann = annotation.symbol

        for p,l in zip(peaks,ann):

            encoded = encode_label(l)

            if encoded is None:
                continue

            if p-window < 0 or p+window+1 > len(signal):
                continue

            beat = signal[p-window:p+window+1]

            beats.append(beat)
            labels.append(encoded)

    return np.array(beats), np.array(labels)

# %%
records = [
'100','101','102','103','104','105','106','107','108','109',
'111','112','113','114','115','116','117','118','119','121',
'122','123','124','200','201','202','203','205','207','208',
'209','210','212','213','214','215','217','219','220','221',
'222','223','228','230','231','232','233','234'
]

folder = "data/mit-bih-arrhythmia-database-1.0.0"

X, y = extract_beats(records, folder)

X = normalize(X)

print(X.shape, y.shape)

X_denoised = []

for beat in tqdm(X):
    clean = denoise(beat)
    X_denoised.append(clean)

X_denoised = np.array(X_denoised)

print("After denoising:", X_denoised.shape)

def normalize(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

X_normalized = normalize(X_denoised)

print("After normalization:", X_normalized.shape)

# %%
import pandas as pd
from sklearn.utils import resample

# Combine X and y
data = np.hstack((X_normalized, y.reshape(-1,1)))

df = pd.DataFrame(data)

label_col = df.shape[1] - 1

# Split classes
df_0 = df[df[label_col] == 0]
df_1 = df[df[label_col] == 1]
df_2 = df[df[label_col] == 2]
df_3 = df[df[label_col] == 3]
df_4 = df[df[label_col] == 4]

target_size = 5000

df_0 = resample(df_0, replace=True, n_samples=target_size, random_state=42)
df_1 = resample(df_1, replace=True, n_samples=target_size, random_state=42)
df_2 = resample(df_2, replace=True, n_samples=target_size, random_state=42)
df_3 = resample(df_3, replace=True, n_samples=target_size, random_state=42)
df_4 = resample(df_4, replace=True, n_samples=target_size, random_state=42)

df_balanced = pd.concat([df_0, df_1, df_2, df_3, df_4])

# # Shuffle
df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

# # Split back
X = df_balanced.iloc[:,:-1].values
y = df_balanced.iloc[:,-1].values

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
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./ecg_model",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=15,
    learning_rate=1e-3,
    logging_steps=50
)


from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "balanced": balanced_accuracy_score(labels, preds),
        "labels": labels,
        "preds": preds
    }

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
print(confusion_matrix(metrics["eval_labels"], metrics["eval_preds"]))


predictions = trainer.predict(test_dataset)

logits = predictions.predictions
labels = predictions.label_ids

preds = logits.argmax(axis=1)

print(classification_report(labels, preds))

print(confusion_matrix(labels, preds))

# %%



