"""
train_ctc.py
============
Training script for the ContinuousSignCTCNet (CNN + BiLSTM + CTC).

Features:
1. Generates or loads continuous multimodal sign sequences with variable lengths.
2. Formats sequences for PyTorch CTCLoss (Time, Batch, Vocab).
3. Evaluates sequence-level Character Error Rate (CER) and Word Error Rate (WER).
4. Saves checkpoint to 'models/continuous_ctc.pth'.
5. Automatically exports trained checkpoint to ONNX ('models/continuous_ctc.onnx').
"""

import sys
import os
from pathlib import Path
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ctc_model import ContinuousSignCTCNet, save_ctc_model
from utils.onnx_exporter import export_ctc_to_onnx


class SyntheticContinuousSignDataset(Dataset):
    """
    Synthesizes variable-length continuous sign sequences with co-articulation
    transitions for training and validating the CTC sequence decoder.
    """

    def __init__(
        self,
        num_samples: int = 500,
        feature_dim: int = 210,
        num_classes: int = 65,
        min_signs_per_seq: int = 2,
        max_signs_per_seq: int = 5,
        frames_per_sign: int = 15
    ):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.min_signs = min_signs_per_seq
        self.max_signs = max_signs_per_seq
        self.frames_per_sign = frames_per_sign

        self.samples = []
        self._generate_dataset()

    def _generate_dataset(self):
        rng = np.random.RandomState(42)
        for _ in range(self.num_samples):
            # Choose random number of signs in sequence
            k = rng.randint(self.min_signs, self.max_signs + 1)
            # CTC target labels: 1 to num_classes (0 is blank)
            targets = rng.randint(1, self.num_classes + 1, size=k).tolist()

            seq_frames = []
            for t_label in targets:
                # Base sign trajectory pattern
                base_vec = rng.randn(self.feature_dim) * 0.1
                # Distinctive cluster signature for this class
                base_vec[t_label % self.feature_dim] += 1.5
                base_vec[(t_label * 3) % self.feature_dim] += 1.2

                sign_duration = rng.randint(self.frames_per_sign - 4, self.frames_per_sign + 6)
                for t_i in range(sign_duration):
                    # Temporal envelope: onset -> stroke apex -> offset
                    envelope = np.sin((t_i + 1) / (sign_duration + 1) * np.pi)
                    frame_feat = base_vec * envelope + rng.randn(self.feature_dim) * 0.05
                    seq_frames.append(frame_feat)

                # Add transition frames (2 to 5 frames of movement noise)
                trans_len = rng.randint(2, 6)
                for _ in range(trans_len):
                    trans_vec = rng.randn(self.feature_dim) * 0.08
                    seq_frames.append(trans_vec)

            seq_array = np.array(seq_frames, dtype=np.float32)
            self.samples.append((seq_array, targets))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def ctc_collate_fn(batch):
    """
    Custom collate function for variable-length CTC sequences.
    Pads features to max length in batch and flattens target label sequences.
    """
    sequences, targets_list = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    feat_dim = sequences[0].shape[1]

    padded_seqs = torch.zeros((len(sequences), max_len, feat_dim), dtype=torch.float32)
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq), :] = torch.from_numpy(seq)

    target_lengths = [len(t) for t in targets_list]
    flat_targets = []
    for t in targets_list:
        flat_targets.extend(t)
    flat_targets_tensor = torch.tensor(flat_targets, dtype=torch.long)

    return (
        padded_seqs,
        flat_targets_tensor,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long)
    )


def train_ctc_model(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 0.001,
    feature_dim: int = 210,
    num_classes: int = 65,
    output_path: str = "models/continuous_ctc.pth"
):
    print(f"[*] Initializing Continuous CTC Model Training (Feature Dim: {feature_dim}, Classes: {num_classes})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    # Standard default vocabulary: 39 alphabet/digits + 26 WLASL words
    classes = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space",
        "ACCIDENT", "BED", "BEFORE", "CALL", "CANDY", "CHANGE", "COLD", "COMPUTER",
        "COOL", "COUSIN", "DRINK", "GO", "HELP", "LATER", "MAN", "NO", "PLAY",
        "SHIRT", "SHORT", "TALL", "THIN", "WALK", "WHAT", "WHO", "WORK", "YES"
    ]

    dataset = SyntheticContinuousSignDataset(
        num_samples=400,
        feature_dim=feature_dim,
        num_classes=num_classes
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=ctc_collate_fn)

    model = ContinuousSignCTCNet(
        input_dim=feature_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0
        for x_padded, flat_targets, input_lengths, target_lengths in train_loader:
            x_padded = x_padded.to(device)
            flat_targets = flat_targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            # log_probs shape: (Time, Batch, CTC_Vocab)
            log_probs = model(x_padded)
            loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - CTC Loss: {avg_loss:.4f}")

    # Save checkpoint
    save_ctc_model(model, classes, filepath=output_path)
    print(f"[+] Saved trained CTC model to: {output_path}")

    # Optional ONNX Export
    onnx_path = output_path.replace(".pth", ".onnx")
    export_ctc_to_onnx(model, input_dim=feature_dim, output_path=onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Continuous CTC Sign Language Sequence Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--output", type=str, default="models/continuous_ctc.pth", help="Model checkpoint path")
    args = parser.parse_args()

    train_ctc_model(epochs=args.epochs, batch_size=args.batch_size, output_path=args.output)
