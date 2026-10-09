"""
Training Pipeline for Deep Learning Motion-Aware Sign Language LSTM.

Trains MotionSignLSTM on temporal sequence windows (N x 20 frames x 129 features).
Uses Stratified 80:20 Train-Test Split, Cross-Entropy Loss, and Adam Optimizer.
Saves:
- models/sign_lstm.pth
- models/label_encoder.pkl
- models/lstm_metadata.json
- data/X_test_seq.npy & data/y_test_seq.npy (Held-out for evaluation)
"""
import os
import sys
import json
import argparse
import datetime
from pathlib import Path
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lstm_model import MotionSignLSTM, save_lstm_checkpoint
from utils.landmarks_util import TOTAL_MOTION_FEATURES, DEFAULT_SEQUENCE_LENGTH


def train_lstm_model(
    x_seq_path: str = "data/X_sequences.npy",
    y_seq_path: str = "data/y_sequences.npy",
    model_output_dir: str = "models",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.002,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.25,
    random_state: int = 42
) -> None:
    """Trains the MotionSignLSTM network on sequence data."""
    x_file = Path(x_seq_path)
    y_file = Path(y_seq_path)
    models_dir = Path(model_output_dir)
    data_dir = x_file.parent

    if not x_file.exists() or not y_file.exists():
        raise FileNotFoundError(
            f"Sequence files '{x_seq_path}' or '{y_seq_path}' not found. "
            "Please run 'python training/generate_sample_data.py' first."
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=======================================================")
    print("      PHASE 2: MOTION-AWARE LSTM MODEL TRAINING        ")
    print("=======================================================")
    print(f"[*] Compute Device        : {device}")

    # Step 1: Load Sequences
    X = np.load(str(x_file))
    y_raw = np.load(str(y_file), allow_pickle=True)

    print(f"[*] Input Data Shape      : {X.shape} (N={X.shape[0]}, T={X.shape[1]}, F={X.shape[2]})")

    # Encode string labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = list(label_encoder.classes_)
    num_classes = len(classes)
    print(f"[*] Distinct ASL Classes  : {num_classes} classes")

    # Step 2: Stratified 80:20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print(f"[*] Train Sequences Count : {len(X_train)}")
    print(f"[*] Test Sequences Count  : {len(X_test)} (Held-out for evaluation)")

    # Save held-out test split for evaluation
    np.save(str(data_dir / "X_test_seq.npy"), X_test)
    np.save(str(data_dir / "y_test_seq.npy"), y_test)
    print(f"[+] Saved held-out test sequences to: {data_dir / 'X_test_seq.npy'}")

    # Convert to PyTorch Tensors
    train_x_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.long)
    test_x_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_y_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 3: Instantiate Model
    model = MotionSignLSTM(
        input_size=X.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n[*] Model Architecture:")
    print(f"    - LSTM Layers : {num_layers} layers (Hidden units: {hidden_size})")
    print(f"    - Parameters  : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"    - Batch Size  : {batch_size} | Epochs: {epochs}")

    # Step 4: Training Loop
    start_time = datetime.datetime.now()
    best_val_acc = 0.0

    print("\n[*] Starting LSTM Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation Step
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_x, val_y in test_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_logits = model(val_x)
                val_preds = torch.argmax(val_logits, dim=1)
                correct_val += (val_preds == val_y).sum().item()
                total_val += val_x.size(0)

        val_acc = correct_val / total_val

        if (epoch % 5 == 0) or (epoch == epochs) or (epoch == 1):
            print(f"    Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_lstm_checkpoint(
                model=model,
                filepath=str(models_dir / "sign_lstm.pth"),
                metadata={
                    "classes": classes,
                    "num_classes": num_classes,
                    "sequence_length": X.shape[1],
                    "features_per_frame": X.shape[2],
                    "best_val_acc": float(best_val_acc),
                    "epoch": epoch
                }
            )

    training_duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n[+] Training finished in {training_duration:.2f}s | Peak Validation Accuracy: {best_val_acc:.2%}")

    # Save Label Encoder
    encoder_path = models_dir / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    print(f"[+] Saved Label Encoder to: {encoder_path}")

    # Save Metadata JSON
    metadata_json = {
        "model_type": "MotionSignLSTM",
        "framework": "PyTorch",
        "classes": classes,
        "num_classes": num_classes,
        "sequence_length": int(X.shape[1]),
        "features_per_frame": int(X.shape[2]),
        "best_validation_accuracy": float(best_val_acc),
        "total_train_samples": len(X_train),
        "total_test_samples": len(X_test),
        "training_duration_seconds": float(training_duration),
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(models_dir / "lstm_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=4)
    print(f"[+] Saved metadata JSON to: {models_dir / 'lstm_metadata.json'}")

    print("\n[+] Phase 2 LSTM Model Training Complete!")
    print("    Next step: Run 'python training/evaluate_lstm.py' to evaluate on unseen test sequences.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Motion-Aware LSTM for ASL Recognition.")
    parser.add_argument("--x-seq", type=str, default="data/X_sequences.npy", help="X sequences .npy path")
    parser.add_argument("--y-seq", type=str, default="data/y_sequences.npy", help="y sequences .npy path")
    parser.add_argument("--models-dir", type=str, default="models", help="Destination folder for models")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    args = parser.parse_args()

    train_lstm_model(
        x_seq_path=args.x_seq,
        y_seq_path=args.y_seq,
        model_output_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
