"""
Train Temporal Motion LSTM on WLASL Word-Level Sequences.
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lstm_model import MotionSignLSTM, save_lstm_checkpoint


def train_wlasl_lstm(
    seq_path: str = "data/wlasl_X_seq.npy",
    y_path: str = "data/wlasl_y.npy",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.002
):
    print("=" * 65)
    print("      TRAINING WLASL TEMPORAL LSTM WORD RECOGNIZER         ")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.load(seq_path)
    y_raw = np.load(y_path)

    encoder = joblib.load("models/wlasl_label_encoder.pkl")
    y_enc = encoder.transform(y_raw)
    num_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MotionSignLSTM(
        input_size=129,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Val
        model.eval()
        corr = 0
        total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                preds = torch.argmax(model(bx), dim=1)
                corr += (preds == by).sum().item()
                total += by.size(0)

        acc = corr / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch:02d}/{epochs:02d}] - Val Acc: {acc:.2%} (Best: {best_acc:.2%})")

    if best_state is not None:
        model.load_state_dict(best_state)
        save_lstm_checkpoint(
            model=model,
            filepath="models/wlasl_word_lstm.pth",
            metadata={
                "classes": list(encoder.classes_),
                "num_classes": num_classes,
                "best_val_acc": float(best_acc)
            }
        )

    print(f"\n[+] WLASL LSTM Word Model Trained! Peak Val Accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    train_wlasl_lstm()
