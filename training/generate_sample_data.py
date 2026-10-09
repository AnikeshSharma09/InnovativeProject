"""
Motion-Aware Sample Landmark Dataset Generator for American Sign Language (ASL).
Generates biologically realistic temporal motion sequences (N x 20 frames x 129 features)
simulating dynamic hand movements, trajectories, and finger articulations
for all standard ASL classes (A-Z, space, del, nothing).

Saves:
- data/X_sequences.npy: 3D NumPy array of shape (N, 20, 129)
- data/y_sequences.npy: 1D array of class labels
- data/landmarks.csv: Flattened CSV for backward compatibility
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.landmarks_util import (
    get_feature_names,
    get_motion_feature_names,
    normalize_landmarks,
    extract_motion_features,
    NUM_LANDMARKS,
    DEFAULT_SEQUENCE_LENGTH,
    TOTAL_MOTION_FEATURES
)
from utils.labels import ALPHABET_LETTERS, SPECIAL_CLASSES


def _build_base_hand(
    thumb_state: str = "curled",
    index_state: str = "curled",
    middle_state: str = "curled",
    ring_state: str = "curled",
    pinky_state: str = "curled",
    wrist_pos: Tuple[float, float, float] = (0.5, 0.8, 0.0)
) -> List[Tuple[float, float, float]]:
    """Constructs a 21-landmark 3D hand model based on anatomical finger extension states."""
    wx, wy, wz = wrist_pos
    
    mcp_roots = {
        "thumb": (wx - 0.08, wy - 0.10, wz - 0.02),
        "index": (wx - 0.04, wy - 0.25, wz - 0.01),
        "middle": (wx, wy - 0.27, wz),
        "ring": (wx + 0.04, wy - 0.25, wz + 0.01),
        "pinky": (wx + 0.08, wy - 0.22, wz + 0.02),
    }

    landmarks: Dict[int, Tuple[float, float, float]] = {0: wrist_pos}

    def add_finger(start_idx: int, root: Tuple[float, float, float], state: str, spread_x: float = 0.0):
        rx, ry, rz = root
        landmarks[start_idx] = root
        
        if state in ["extended", "open", "spread"]:
            landmarks[start_idx + 1] = (rx + spread_x * 0.3, ry - 0.08, rz - 0.01)
            landmarks[start_idx + 2] = (rx + spread_x * 0.7, ry - 0.16, rz - 0.02)
            landmarks[start_idx + 3] = (rx + spread_x * 1.0, ry - 0.24, rz - 0.03)
        elif state == "curled":
            landmarks[start_idx + 1] = (rx, ry - 0.05, rz + 0.04)
            landmarks[start_idx + 2] = (rx, ry - 0.02, rz + 0.07)
            landmarks[start_idx + 3] = (rx, ry + 0.02, rz + 0.05)
        elif state == "hooked":
            landmarks[start_idx + 1] = (rx, ry - 0.06, rz + 0.02)
            landmarks[start_idx + 2] = (rx, ry - 0.08, rz + 0.06)
            landmarks[start_idx + 3] = (rx, ry - 0.05, rz + 0.08)

    # 1-4: Thumb
    if thumb_state == "extended":
        landmarks[1] = mcp_roots["thumb"]
        landmarks[2] = (wx - 0.14, wy - 0.15, wz - 0.03)
        landmarks[3] = (wx - 0.19, wy - 0.22, wz - 0.04)
        landmarks[4] = (wx - 0.24, wy - 0.28, wz - 0.05)
    elif thumb_state == "up":
        landmarks[1] = mcp_roots["thumb"]
        landmarks[2] = (wx - 0.12, wy - 0.18, wz - 0.02)
        landmarks[3] = (wx - 0.14, wy - 0.27, wz - 0.02)
        landmarks[4] = (wx - 0.15, wy - 0.35, wz - 0.03)
    elif thumb_state == "across":
        landmarks[1] = mcp_roots["thumb"]
        landmarks[2] = (wx - 0.06, wy - 0.14, wz + 0.02)
        landmarks[3] = (wx - 0.02, wy - 0.17, wz + 0.04)
        landmarks[4] = (wx + 0.02, wy - 0.19, wz + 0.05)
    else:  # curled
        landmarks[1] = mcp_roots["thumb"]
        landmarks[2] = (wx - 0.10, wy - 0.12, wz + 0.02)
        landmarks[3] = (wx - 0.08, wy - 0.16, wz + 0.03)
        landmarks[4] = (wx - 0.06, wy - 0.20, wz + 0.04)

    add_finger(5, mcp_roots["index"], index_state, spread_x=-0.03)
    add_finger(9, mcp_roots["middle"], middle_state, spread_x=0.0)
    add_finger(13, mcp_roots["ring"], ring_state, spread_x=0.02)
    add_finger(17, mcp_roots["pinky"], pinky_state, spread_x=0.05)

    return [landmarks[i] for i in range(NUM_LANDMARKS)]


def get_template_for_class(cls_name: str, t_ratio: float = 0.0) -> List[Tuple[float, float, float]]:
    """
    Returns the anatomical hand configuration for class at time progression t_ratio in [0, 1].
    Handles dynamic motions for 'J' (tracing hook), 'Z' (tracing zigzag), 'space' (sweep), etc.
    """
    c = cls_name.upper() if cls_name in ALPHABET_LETTERS else cls_name.lower()
    
    # Dynamic Sign: J (pinky extended, whole hand traces a J curve)
    if c == "J":
        # Trajectory: straight down then curve up-left
        if t_ratio < 0.6:
            wx = 0.50
            wy = 0.70 + (t_ratio / 0.6) * 0.12
        else:
            angle = ((t_ratio - 0.6) / 0.4) * np.pi
            wx = 0.50 - 0.08 * np.sin(angle)
            wy = 0.82 - 0.06 * (1.0 - np.cos(angle))
        return _build_base_hand(thumb_state="across", index_state="curled", middle_state="curled", ring_state="curled", pinky_state="extended", wrist_pos=(wx, wy, 0.0))

    # Dynamic Sign: Z (index finger extended, traces Z zigzag path)
    elif c == "Z":
        if t_ratio < 0.33:  # Horizontal right
            wx = 0.45 + (t_ratio / 0.33) * 0.10
            wy = 0.70
        elif t_ratio < 0.66:  # Diagonal down-left
            r = (t_ratio - 0.33) / 0.33
            wx = 0.55 - r * 0.10
            wy = 0.70 + r * 0.10
        else:  # Horizontal right
            r = (t_ratio - 0.66) / 0.34
            wx = 0.45 + r * 0.10
            wy = 0.80
        return _build_base_hand(thumb_state="across", index_state="extended", middle_state="curled", ring_state="curled", pinky_state="curled", wrist_pos=(wx, wy, 0.0))

    # Dynamic Sign: Space (open hand sweeps horizontally across camera)
    elif c == "space":
        wx = 0.40 + t_ratio * 0.20
        wy = 0.78
        return _build_base_hand(thumb_state="open", index_state="open", middle_state="open", ring_state="open", pinky_state="open", wrist_pos=(wx, wy, 0.0))

    # Dynamic Sign: Del (waving back / flicking backwards)
    elif c == "del":
        wx = 0.60 - t_ratio * 0.15
        wy = 0.76
        return _build_base_hand(thumb_state="up", index_state="extended", middle_state="extended", ring_state="extended", pinky_state="curled", wrist_pos=(wx, wy, 0.0))

    # Static signs with subtle human micro-motion
    elif c == "A":
        return _build_base_hand(thumb_state="up", index_state="curled", middle_state="curled", ring_state="curled", pinky_state="curled")
    elif c == "B":
        return _build_base_hand(thumb_state="across", index_state="extended", middle_state="extended", ring_state="extended", pinky_state="extended")
    elif c == "C":
        return _build_base_hand(thumb_state="extended", index_state="hooked", middle_state="hooked", ring_state="hooked", pinky_state="hooked")
    elif c == "D":
        return _build_base_hand(thumb_state="across", index_state="extended", middle_state="curled", ring_state="curled", pinky_state="curled")
    elif c == "E":
        return _build_base_hand(thumb_state="across", index_state="hooked", middle_state="hooked", ring_state="hooked", pinky_state="hooked")
    elif c == "F":
        return _build_base_hand(thumb_state="across", index_state="curled", middle_state="extended", ring_state="extended", pinky_state="extended")
    elif c == "G":
        return _build_base_hand(thumb_state="extended", index_state="extended", middle_state="curled", ring_state="curled", pinky_state="curled")
    elif c == "H":
        return _build_base_hand(thumb_state="curled", index_state="extended", middle_state="extended", ring_state="curled", pinky_state="curled")
    elif c == "I":
        return _build_base_hand(thumb_state="across", index_state="curled", middle_state="curled", ring_state="curled", pinky_state="extended")
    elif c == "K":
        return _build_base_hand(thumb_state="up", index_state="extended", middle_state="extended", ring_state="curled", pinky_state="curled")
    elif c == "L":
        return _build_base_hand(thumb_state="extended", index_state="extended", middle_state="curled", ring_state="curled", pinky_state="curled")
    elif c == "O":
        return _build_base_hand(thumb_state="across", index_state="hooked", middle_state="hooked", ring_state="hooked", pinky_state="hooked")
    elif c == "U":
        return _build_base_hand(thumb_state="across", index_state="extended", middle_state="extended", ring_state="curled", pinky_state="curled")
    elif c == "V":
        return _build_base_hand(thumb_state="across", index_state="spread", middle_state="spread", ring_state="curled", pinky_state="curled")
    elif c == "W":
        return _build_base_hand(thumb_state="across", index_state="spread", middle_state="spread", ring_state="spread", pinky_state="curled")
    elif c == "Y":
        return _build_base_hand(thumb_state="extended", index_state="curled", middle_state="curled", ring_state="curled", pinky_state="extended")
    elif c == "nothing":
        return _build_base_hand(thumb_state="curled", index_state="curled", middle_state="curled", ring_state="curled", pinky_state="curled")
    else:
        return _build_base_hand(thumb_state="curled", index_state="curled", middle_state="curled", ring_state="curled", pinky_state="curled")


def generate_motion_dataset(
    output_dir: str = "data",
    samples_per_class: int = 70,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    noise_level: float = 0.015,
    classes: List[str] = None
) -> Tuple[str, str]:
    """
    Generates temporal motion sequences:
    - X_sequences.npy: shape (N, sequence_length=20, 129 features)
    - y_sequences.npy: shape (N,) labels
    - landmarks.csv: flattened tabular dataset for backward compatibility
    """
    if classes is None:
        classes = sorted(ALPHABET_LETTERS + SPECIAL_CLASSES)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"      GENERATING MOTION-AWARE LSTM SEQUENCES DATASET   ")
    print(f"=======================================================")
    print(f"[*] Distinct Classes      : {len(classes)}")
    print(f"[*] Sequences Per Class   : {samples_per_class}")
    print(f"[*] Total Sequences       : {len(classes) * samples_per_class}")
    print(f"[*] Sequence Length (T)   : {sequence_length} frames")
    print(f"[*] Features Per Frame    : {TOTAL_MOTION_FEATURES} (63 spatial + 63 vel + 3 wrist)")

    rng = np.random.default_rng(seed=42)
    all_sequences = []
    all_labels = []
    csv_records = []
    feature_names_63 = get_feature_names()

    for cls_name in classes:
        for s_idx in range(samples_per_class):
            sequence_frames = []
            prev_coords = None

            # Add random micro-shift to the sample's starting position
            shift_x = float(rng.normal(0, 0.02))
            shift_y = float(rng.normal(0, 0.02))

            for t_step in range(sequence_length):
                t_ratio = t_step / max(1, sequence_length - 1)
                base_coords = get_template_for_class(cls_name, t_ratio)

                # Add noise and hand jitter across time
                noisy_coords = []
                for (x, y, z) in base_coords:
                    nx = float(x + shift_x + rng.normal(0, noise_level))
                    ny = float(y + shift_y + rng.normal(0, noise_level))
                    nz = float(z + rng.normal(0, noise_level * 0.5))
                    noisy_coords.append((nx, ny, nz))

                # Extract 129 motion features
                motion_vec, ke = extract_motion_features(noisy_coords, prev_coords)
                sequence_frames.append(motion_vec)
                prev_coords = noisy_coords

                # For the CSV dataset, store the final frame's 63 spatial features
                if t_step == sequence_length - 1:
                    row = dict(zip(feature_names_63, motion_vec[:63]))
                    row["label"] = cls_name
                    csv_records.append(row)

            all_sequences.append(sequence_frames)
            all_labels.append(cls_name)

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=object)

    x_seq_file = out_path / "X_sequences.npy"
    y_seq_file = out_path / "y_sequences.npy"
    csv_file = out_path / "landmarks.csv"

    np.save(str(x_seq_file), X)
    np.save(str(y_seq_file), y)
    pd.DataFrame(csv_records).to_csv(csv_file, index=False)

    print(f"\n[+] SUCCESS:")
    print(f"    - Motion Sequences saved : {x_seq_file} (Shape: {X.shape})")
    print(f"    - Sequence Labels saved  : {y_seq_file} (Shape: {y.shape})")
    print(f"    - Tabular CSV saved      : {csv_file} ({len(csv_records)} rows)")

    return str(x_seq_file), str(y_seq_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate motion-aware LSTM sequences dataset.")
    parser.add_argument("--output-dir", type=str, default="data", help="Target output folder")
    parser.add_argument("--samples-per-class", type=int, default=70, help="Number of sequences per class")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH, help="Frames per sequence")
    args = parser.parse_args()

    generate_motion_dataset(
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        sequence_length=args.sequence_length
    )
