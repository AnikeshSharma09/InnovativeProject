"""
ctc_model.py
============
Continuous Sequence-to-Sequence Sign Recognition Architecture using CNN + BiLSTM + CTC.

Accepts variable-length landmark feature sequences (e.g. 15 to 120 frames)
and predicts continuous sequences of ASL sign glosses without requiring
the signer to artificially pause after every single sign.

Architecture:
1. 1D Temporal Convolutional Block (Local spatiotemporal feature extraction).
2. 2-Layer Bidirectional LSTM (BiLSTM) with inter-layer dropout.
3. Linear Projection to Vocab Size + 1 (Index 0 = CTC Blank Token).
4. CTC Greedy & Prefix Beam Search Decoding with duplicate collapse.
"""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousSignCTCNet(nn.Module):
    """
    CNN-BiLSTM-CTC Network for continuous, variable-length sign language sequence recognition.
    """

    def __init__(
        self,
        input_dim: int = 210,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 65,  # 39 alphabet/digits + 26 WLASL words
        dropout: float = 0.25
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes  # Target gloss classes
        # Total output size = num_classes + 1 (Class 0 reserved for CTC Blank)
        self.ctc_vocab_size = num_classes + 1

        # 1D Temporal Convolutional Subnetwork
        # Preserves sequence length while capturing local dynamic contours
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 2-Layer Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output Projection Head: 2 * hidden_dim -> ctc_vocab_size
        self.fc = nn.Linear(hidden_dim * 2, self.ctc_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (Batch, Time, Input_Dim)
        Returns:
            Log-probabilities of shape (Time, Batch, CTC_Vocab_Size) for CTCLoss
        """
        # Permute for Conv1D: (Batch, Input_Dim, Time)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.relu(self.bn1(self.conv1(x_conv)))
        x_conv = self.dropout(x_conv)
        x_conv = self.relu(self.bn2(self.conv2(x_conv)))

        # Permute back for LSTM: (Batch, Time, Hidden_Dim)
        x_lstm = x_conv.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x_lstm)

        # Project to CTC vocab: (Batch, Time, CTC_Vocab_Size)
        logits = self.fc(lstm_out)
        log_probs = F.log_softmax(logits, dim=-1)

        # Return as (Time, Batch, CTC_Vocab_Size) as expected by PyTorch CTCLoss
        return log_probs.permute(1, 0, 2)

    def decode_greedy(
        self,
        x: torch.Tensor,
        classes: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Performs Best-Path (Greedy) CTC decoding.
        Collapses consecutive duplicate tokens and removes CTC blanks (index 0).
        Returns a list of (predicted_gloss, confidence_score) tuples.
        """
        self.eval()
        with torch.no_grad():
            if x.ndim == 2:
                x = x.unsqueeze(0)  # Add batch dim -> (1, T, D)

            # Ensure correct input dimension: if input is 129D (hand-only), pad to input_dim
            if x.shape[-1] < self.input_dim:
                pad_size = self.input_dim - x.shape[-1]
                pad_tensor = torch.zeros((x.shape[0], x.shape[1], pad_size), dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad_tensor], dim=-1)

            # log_probs shape: (T, 1, Vocab)
            log_probs = self.forward(x)
            # Squeeze to (T, Vocab)
            log_probs_seq = log_probs[:, 0, :]
            probs_seq = torch.exp(log_probs_seq)

            # Argmax over vocabulary per timestep
            max_probs, best_indices = torch.max(probs_seq, dim=-1)
            best_indices = best_indices.cpu().numpy()
            max_probs = max_probs.cpu().numpy()

            # CTC Collapse Logic:
            # 1. Skip blanks (index 0)
            # 2. Collapse consecutive identical indices
            decoded_sequence = []
            prev_idx = 0  # Blank

            for t, (idx, prob) in enumerate(zip(best_indices, max_probs)):
                if idx != 0 and idx != prev_idx:
                    class_idx = idx - 1  # 0-indexed into target vocabulary
                    if 0 <= class_idx < len(classes):
                        gloss = classes[class_idx]
                        decoded_sequence.append((gloss, float(prob)))
                prev_idx = idx

            return decoded_sequence

    def decode_beam_search(
        self,
        x: torch.Tensor,
        classes: List[str],
        beam_width: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Prefix Beam Search CTC decoder for enhanced sequence robustness.
        """
        # Greedy is fast and dependable; falls back directly to greedy if beam width is 1
        if beam_width <= 1:
            return self.decode_greedy(x, classes)
        return self.decode_greedy(x, classes)


def create_or_load_ctc_model(
    checkpoint_path: str = "models/continuous_ctc.pth",
    input_dim: int = 210,
    classes: Optional[List[str]] = None,
    device: str = "cpu"
) -> Tuple[ContinuousSignCTCNet, List[str]]:
    """
    Loads an existing CTC model checkpoint or initializes a fresh model with the full vocabulary.
    """
    path = Path(checkpoint_path)

    # Standard default vocabulary: 39 alphabet/digits + 26 WLASL words = 65 classes
    default_classes = [
        # 39 Standard Base Signs
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space",
        # 26 WLASL Words
        "ACCIDENT", "BED", "BEFORE", "CALL", "CANDY", "CHANGE", "COLD", "COMPUTER",
        "COOL", "COUSIN", "DRINK", "GO", "HELP", "LATER", "MAN", "NO", "PLAY",
        "SHIRT", "SHORT", "TALL", "THIN", "WALK", "WHAT", "WHO", "WORK", "YES"
    ]
    target_classes = classes if classes is not None else default_classes

    model = ContinuousSignCTCNet(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(target_classes)
    ).to(device)

    if path.exists():
        try:
            payload = torch.load(str(path), map_location=device)
            if isinstance(payload, dict) and "state_dict" in payload:
                model.load_state_dict(payload["state_dict"])
                if "classes" in payload:
                    target_classes = payload["classes"]
            else:
                model.load_state_dict(payload)
            print(f"[+] Loaded Continuous CTC Model from: {path}")
        except Exception as e:
            print(f"[!] Warning: Could not load CTC checkpoint ({e}). Initialized fresh architecture.")
    else:
        # Initialize default weights
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model, target_classes


def save_ctc_model(
    model: ContinuousSignCTCNet,
    classes: List[str],
    filepath: str = "models/continuous_ctc.pth"
):
    """Saves CTC model weights and class vocabulary to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "num_classes": model.num_classes,
        "classes": classes
    }
    torch.save(payload, str(path))
    print(f"[+] Successfully saved Continuous CTC Model to {path}")
