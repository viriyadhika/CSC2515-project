import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wfdb
from imblearn.over_sampling import SMOTE



# import YOUR dataset loader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IDX2CLS = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

window = 93


def encode_label(l):

    if l in ['N','L','R','e','j']:
        return 0   # Normal

    elif l in ['A','a','J','S']:
        return 1   # Supraventricular ectopic

    elif l in ['V','E']:
        return 2   # Ventricular ectopic

    elif l in ['F']:
        return 3   # Fusion beat

    elif l in ['/', 'f', 'Q']:
        return 4   # Unknown / paced

    return None


def extract_beats(record_list, folder):

    beats = []
    labels = []

    for r in record_list:

        path = f"{folder}/{r}"

        record = wfdb.rdrecord(path)
        annotation = wfdb.rdann(path, 'atr')

        signal = record.p_signal[:, 0]
        peaks = annotation.sample
        ann = annotation.symbol

        for p, l in zip(peaks, ann):

            encoded = encode_label(l)

            if encoded is None:
                continue

            if p-window < 0 or p+window+1 > len(signal):
                continue

            beat = signal[p-window:p+window+1]

            beats.append(beat)
            labels.append(encoded)

    return np.array(beats), np.array(labels)


def preprocess_ecg(X_train, y_train, X_test, apply_noise=False):

    # ----------------------------
    # remove NaNs
    # ----------------------------
    mask = ~np.isnan(X_train).any(axis=1)

    X_train = X_train[mask]
    y_train = y_train[mask]

    # ----------------------------
    # peak enhancement
    # ----------------------------
    X_train = X_train / np.max(np.abs(X_train), axis=1, keepdims=True)
    X_test = X_test / np.max(np.abs(X_test), axis=1, keepdims=True)

    # ----------------------------
    # optional Gaussian noise (train only)
    # ----------------------------
    if apply_noise:

        noise = np.random.normal(0, 0.05, X_train.shape)

        X_train = X_train + noise

    # ----------------------------
    # SMOTE balancing (train only)
    # ----------------------------
    smote = SMOTE()

    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, y_train, X_test

class InterPatientDataset:

    def __init__(self, records, folder, test_size=0.2, random_state=42):

        train_records, test_records = train_test_split(
            records,
            test_size=test_size,
            random_state=random_state
        )

        X_train, y_train = extract_beats(train_records, folder)
        X_test, y_test = extract_beats(test_records, folder)

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        self.X_train = (X_train - mean) / std
        self.X_test = (X_test - mean) / std

        self.y_train = y_train
        self.y_test = y_test

        print("Inter-patient split")
        print("Train shape:", self.X_train.shape)
        print("Test shape:", self.X_test.shape)

class IntraPatientDataset:
    def __init__(self, records, folder, test_size=0.2, random_state=42):

        X, y = extract_beats(records, folder)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        self.X_train = X_train
        self.X_test = X_test
        
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        
        self.X_train = (X_train - mean) / std
        self.X_test = (X_test - mean) / std


        self.y_train = y_train
        self.y_test = y_test

        print("Intra-patient split")
        print("Train shape:", self.X_train.shape)
        print("Test shape:", self.X_test.shape)

# -------------------------------------------------------
# Torch Dataset
# -------------------------------------------------------

class TorchBeatDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        x = self.X[idx].unsqueeze(0)
        y = self.y[idx]

        return x, y


# -------------------------------------------------------
# Models
# -------------------------------------------------------

class Yildirim2018ApproxCNN(nn.Module):

    def __init__(self, n_classes=5):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(1,32,7,padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32,64,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64,128,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128,256,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(256*32,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,n_classes)
        )

    def forward(self,x):

        x = self.features(x)

        return self.classifier(x)


class CNNBiLSTM9152186(nn.Module):

    def __init__(self,n_classes=5):

        super().__init__()

        self.conv1 = nn.Conv1d(1,32,5)
        self.conv2 = nn.Conv1d(32,32,5)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        self.lstm1 = nn.LSTM(
            input_size=32,
            hidden_size=32,
            bidirectional=True,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            bidirectional=True,
            batch_first=True
        )

        self.fc1 = nn.Linear(64,16)
        self.fc2 = nn.Linear(16,n_classes)

    def forward(self,x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.transpose(1,2)

        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)

        x = x[:,-1,:]

        x = self.relu(self.fc1(x))

        return self.fc2(x)


class TinyTransformer2402(nn.Module):

    def __init__(self,n_classes=5):

        super().__init__()

        self.patch = nn.Conv1d(
            1,
            16,
            kernel_size=31,
            stride=3,
            padding=15
        )

        encoder = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=8,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder,1)

        self.norm = nn.LayerNorm(16)

        self.head = nn.Linear(16,n_classes)

    def forward(self,x):

        x = self.patch(x)

        x = x.transpose(1,2)

        x = self.transformer(x)

        x = self.norm(x)

        x = x.mean(dim=1)

        return self.head(x)


# -------------------------------------------------------
# Training
# -------------------------------------------------------

def train_model(model,train_loader,test_loader,device,epochs,lr):

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):

        model.train()

        losses = []

        for x,y in train_loader:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = loss_fn(pred,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print("epoch",epoch,"loss",np.mean(losses))

        evaluate(model,test_loader,device)


def evaluate(model,loader,device):

    model.eval()

    y_true=[]
    y_pred=[]

    with torch.no_grad():

        for x,y in loader:

            x = x.to(device)

            pred = model(x).argmax(1).cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(y.numpy())

    print("Accuracy:",accuracy_score(y_true,y_pred))
    print("Balanced Accuracy", balanced_accuracy_score(y_true, y_pred))
    print("Macro F1:",f1_score(y_true,y_pred,average="macro"))

    print(classification_report(
        y_true,
        y_pred,
        target_names=[IDX2CLS[i] for i in range(5)]
    ))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix")
    print(cm)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model",required=True,choices=[
        "cnn",
        "cnn_bilstm",
        "transformer"
    ])

    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--lr",type=float,default=1e-3)

    args = parser.parse_args()

    folder = "mit-bih-arrhythmia-database-1.0.0"

    records = [
        f.replace(".hea","")
        for f in os.listdir(folder)
        if f.endswith(".hea")
    ]

    ds = InterPatientDataset(records,folder)

    X_train, X_test = ds.X_train, ds.X_test
    y_train, y_test = ds.y_train, ds.y_test

    X_train, y_train, X_test = preprocess_ecg(
        X_train,
        y_train,
        X_test,
        apply_noise=True
    )


    train_ds = TorchBeatDataset(X_train,y_train)
    test_ds = TorchBeatDataset(X_test,y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "cnn":
        model = Yildirim2018ApproxCNN()

    elif args.model == "cnn_bilstm":
        model = CNNBiLSTM9152186()

    else:
        model = TinyTransformer2402()

    train_model(
        model,
        train_loader,
        test_loader,
        device,
        args.epochs,
        args.lr
    )


if __name__ == "__main__":
    main()