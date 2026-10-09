"""
Deep Learning LSTM Architecture for Motion-Aware Sign Language Detection.

Models temporal hand dynamics, finger velocity, and spatial trajectories
over a sequence window (e.g. 20 consecutive frames).
Input per frame: 129 features (63 spatial + 63 velocity + 3 global trajectory).
"""
import os
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionSignLSTM(nn.Module):
    """
    Recurrent Neural Network with 2-layer Long Short-Term Memory (LSTM) cells
    and a deep classification head for motion-based sign recognition.
    """

    def __init__(
        self,
        input_size: int = 129,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 29,
        dropout: float = 0.25
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 2-Layer LSTM with dropout between recurrent layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Classification Head
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # lstm_out: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Aggregate temporal sequence: extract the final recurrent step
        last_step = lstm_out[:, -1, :]

        # Fully connected projection
        dense = self.fc1(last_step)
        dense = self.relu(dense)
        dense = self.dropout(dense)
        logits = self.fc2(dense)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Computes normalized class probabilities via Softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities


def save_lstm_checkpoint(
    model: MotionSignLSTM,
    filepath: str = "models/sign_lstm.pth",
    metadata: Optional[dict] = None
) -> None:
    """Saves model weights and configuration to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_classes": model.num_classes,
        "metadata": metadata or {}
    }
    torch.save(payload, str(path))
    print(f"[+] Saved LSTM checkpoint to: {path}")


def load_lstm_checkpoint(
    filepath: str = "models/sign_lstm.pth",
    device: str = "cpu"
) -> Tuple[MotionSignLSTM, dict]:
    """Loads a saved LSTM model checkpoint and instantiates the architecture."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"LSTM model checkpoint not found at: {filepath}")

    payload = torch.load(str(path), map_location=device)
    model = MotionSignLSTM(
        input_size=payload.get("input_size", 129),
        hidden_size=payload.get("hidden_size", 128),
        num_layers=payload.get("num_layers", 2),
        num_classes=payload.get("num_classes", 29)
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload.get("metadata", {})
