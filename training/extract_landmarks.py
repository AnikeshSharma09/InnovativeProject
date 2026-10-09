"""
Phase 1: Dataset Preparation - Real ASL Alphabet Landmark & Motion Sequence Extraction.

Scans the real ASL Alphabet dataset (e.g. archive/asl_alphabet_train/asl_alphabet_train/),
processes real images using OpenCV and MediaPipe Hands:
- Extracts 21 3D hand landmarks and relative normalized coordinates.
- Computes 129-dimensional motion vectors across sequential frames:
  (63 spatial + 63 joint velocities + 3 wrist trajectory deltas).
- Builds rolling sequence windows of length T=20 frames for the PyTorch LSTM network.
- Exports tabular data to data/landmarks.csv.
- Exports motion sequences to data/X_sequences.npy and data/y_sequences.npy.
- Logs skipped / no-hand images to data/extraction_log.txt.
"""
import os
import sys
import argparse
import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.landmarks_util import (
    extract_landmarks_from_bgr_frame,
    robust_extract_landmarks_from_bgr_frame,
    extract_motion_features,
    get_feature_names,
    get_motion_feature_names,
    TOTAL_SPATIAL_FEATURES,
    TOTAL_MOTION_FEATURES,
    DEFAULT_SEQUENCE_LENGTH,
    NUM_LANDMARKS
)
from utils.labels import DEFAULT_ASL_CLASSES


def find_dataset_directory(preferred_dir: Optional[str] = None) -> Path:
    """Auto-detects the ASL dataset directory across standard locations."""
    candidates = [
        preferred_dir,
        "dataset",
        "archive/asl_alphabet_train/asl_alphabet_train",
        "archive/asl_alphabet_train",
        "archive",
        "data/dataset"
    ]
    for c in candidates:
        if c:
            p = Path(c)
            if p.exists() and p.is_dir():
                subdirs = [d.name for d in p.iterdir() if d.is_dir()]
                if any(k in subdirs for k in ["A", "B", "C", "0", "1"]):
                    return p
                for child in p.iterdir():
                    if child.is_dir():
                        nested = [d.name for d in child.iterdir() if d.is_dir()]
                        if any(k in nested for k in ["A", "B", "C", "0", "1"]):
                            return child
    return Path("dataset")


def extract_landmarks_from_dataset(
    dataset_dir: Optional[str] = None,
    output_dir: str = "data",
    max_images_per_class: Optional[int] = 200,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
) -> Tuple[str, str, str]:
    """
    Extracts tabular landmarks and motion sequences from the real ASL dataset.
    """
    dataset_path = find_dataset_directory(dataset_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "landmarks.csv"
    x_seq_path = out_dir / "X_sequences.npy"
    y_seq_path = out_dir / "y_sequences.npy"
    log_path = out_dir / "extraction_log.txt"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at '{dataset_path}'. "
            "Please ensure the 'archive' or 'dataset' folder exists."
        )

    # Discover class subdirectories (excluding any nested dataset containers)
    subdirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name.lower() != "asl_dataset"])
    if not subdirs:
        raise ValueError(f"No class folders found in '{dataset_path}'.")

    print("\n=======================================================")
    print("      EXTRACTING REAL LANDMARKS & MOTION SEQUENCES     ")
    print("=======================================================")
    print(f"[*] Dataset Source Directory : {dataset_path}")
    print(f"[*] Distinct Class Folders   : {len(subdirs)}")
    print(f"[*] Max Images Per Class     : {max_images_per_class if max_images_per_class else 'ALL (~3,000/class)'}")
    print(f"[*] Sequence Window (T)      : {sequence_length} frames")
    print(f"[*] Features Per Frame       : {TOTAL_MOTION_FEATURES} (63 spatial + 63 velocity + 3 trajectory)")

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.50
    )

    feature_names_63 = get_feature_names()
    tabular_records = []
    all_motion_sequences = []
    all_sequence_labels = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    skipped_count = 0
    total_images_processed = 0

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"=== Extraction Started: {datetime.datetime.now().isoformat()} ===\n")

        for class_dir in subdirs:
            class_name = class_dir.name
            image_files = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in valid_extensions])

            if max_images_per_class:
                image_files = image_files[:max_images_per_class]

            print(f"[*] Processing '{class_name}' ({len(image_files)} images)...", end="", flush=True)

            class_features_buffer: List[List[float]] = []
            prev_raw_coords: Optional[List[Tuple[float, float, float]]] = None
            detected_in_class = 0

            for img_path in image_files:
                total_images_processed += 1
                bgr_img = cv2.imread(str(img_path))

                if bgr_img is None:
                    skipped_count += 1
                    log.write(f"UNREADABLE: {img_path}\n")
                    continue

                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                rgb_img.flags.writeable = False
                res = hands_detector.process(rgb_img)

                curr_raw = None
                if res.multi_hand_landmarks:
                    primary_hand = res.multi_hand_landmarks[0]
                    curr_raw = [(lm.x, lm.y, lm.z) for lm in primary_hand.landmark]
                else:
                    # Robust fallback for tightly cropped Kaggle dataset images
                    _, _, robust_coords, _ = robust_extract_landmarks_from_bgr_frame(bgr_img)
                    if robust_coords is not None:
                        curr_raw = robust_coords

                if class_name == "nothing":
                    # Negative control class (no hand present in background)
                    zero_motion = [0.0] * TOTAL_MOTION_FEATURES
                    class_features_buffer.append(zero_motion)
                    spatial_row = dict(zip(feature_names_63, [0.0] * TOTAL_SPATIAL_FEATURES))
                    spatial_row["label"] = class_name
                    tabular_records.append(spatial_row)
                    detected_in_class += 1
                    continue

                if curr_raw is not None:
                    # Extract 129 motion features (spatial + velocity + wrist trajectory)
                    motion_vec, _ = extract_motion_features(curr_raw, prev_raw_coords)
                    prev_raw_coords = curr_raw
                    class_features_buffer.append(motion_vec)

                    # Also save tabular spatial features (63) for Random Forest baseline
                    spatial_row = dict(zip(feature_names_63, motion_vec[:TOTAL_SPATIAL_FEATURES]))
                    spatial_row["label"] = class_name
                    tabular_records.append(spatial_row)

                    detected_in_class += 1
                else:
                    skipped_count += 1
                    prev_raw_coords = None
                    log.write(f"NO_HAND_DETECTED: {img_path}\n")

            # 1. Overlapping sequential slices (stride = 2 for high sequence coverage)
            step_stride = 2
            num_seqs_for_class = 0

            if len(class_features_buffer) >= sequence_length:
                for start_idx in range(0, len(class_features_buffer) - sequence_length + 1, step_stride):
                    seq_slice = class_features_buffer[start_idx : start_idx + sequence_length]
                    all_motion_sequences.append(seq_slice)
                    all_sequence_labels.append(class_name)
                    num_seqs_for_class += 1

            # 2. Realistic static-hold sequences (simulates human holding a sign with micro-jitter)
            np.random.seed(42)
            for m_vec in class_features_buffer:
                if class_name == "nothing":
                    hold_seq = [[0.0] * TOTAL_MOTION_FEATURES for _ in range(sequence_length)]
                else:
                    spatial_part = np.array(m_vec[:TOTAL_SPATIAL_FEATURES], dtype=np.float32)
                    hold_seq = []
                    prev_coords = spatial_part.copy()
                    for f_idx in range(sequence_length):
                        jitter = np.random.normal(0.0, 0.003, size=TOTAL_SPATIAL_FEATURES).astype(np.float32)
                        curr_coords = spatial_part + jitter if f_idx > 0 else spatial_part
                        vel = curr_coords - prev_coords
                        prev_coords = curr_coords
                        frame_vec = list(curr_coords) + list(vel) + [float(vel[0]), float(vel[1]), float(vel[2])]
                        hold_seq.append(frame_vec)
                all_motion_sequences.append(hold_seq)
                all_sequence_labels.append(class_name)
                num_seqs_for_class += 1

            print(f" -> {detected_in_class} hands detected | {num_seqs_for_class} motion sequences built.")

        log.write(f"=== Extraction Finished: {len(tabular_records)} images valid, {skipped_count} skipped ===\n")

    hands_detector.close()

    # Save Tabular CSV
    df = pd.DataFrame(tabular_records)
    df.to_csv(csv_path, index=False)

    # Save Sequence Arrays
    X_seq = np.array(all_motion_sequences, dtype=np.float32)
    y_seq = np.array(all_sequence_labels, dtype=object)
    np.save(str(x_seq_path), X_seq)
    np.save(str(y_seq_path), y_seq)

    print(f"\n[+] EXTRACTION COMPLETE:")
    print(f"    - Total Valid Images Extracted : {len(df):,}")
    print(f"    - Skipped / No Hand Images     : {skipped_count:,}")
    print(f"    - Tabular CSV Saved            : {csv_path} ({len(df)} rows)")
    print(f"    - Motion Sequences Saved       : {x_seq_path} (Shape: {X_seq.shape})")
    print(f"    - Sequence Labels Saved        : {y_seq_path} (Shape: {y_seq.shape})")

    return str(csv_path), str(x_seq_path), str(y_seq_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract landmarks & motion sequences from real ASL dataset.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to ASL image dataset")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--max-per-class", type=int, default=200, help="Max images per class (default: 200)")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH, help="Sequence length T")
    args = parser.parse_args()

    extract_landmarks_from_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_images_per_class=args.max_per_class,
        sequence_length=args.sequence_length
    )
