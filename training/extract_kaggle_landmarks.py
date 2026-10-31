"""
Extracts diverse, high-quality landmarks from real Kaggle ASL Alphabet Train dataset (29 classes)
plus digit classes (0-9) to produce data/landmarks_v2.csv.
"""
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.landmarks_util import (
    robust_extract_landmarks_from_bgr_frame,
    get_feature_names,
    TOTAL_SPATIAL_FEATURES
)

def main():
    print("=" * 65)
    print("  EXTRACTING BALANCED MULTI-SIGNER ASL LANDMARK DATASET (V2)  ")
    print("=" * 65)

    alphabet_dir = Path("archive/asl_alphabet_train/asl_alphabet_train")
    digit_dir = Path("dataset")
    out_csv = Path("data/landmarks_v2.csv")

    feature_cols = get_feature_names()
    records = []

    # 1. Process 29 classes from Kaggle ASL Alphabet (100 images per class)
    if alphabet_dir.exists():
        classes_29 = sorted([d.name for d in alphabet_dir.iterdir() if d.is_dir()])
        print(f"[*] Processing 29 Kaggle Alphabet classes from {alphabet_dir}...")
        for cls in classes_29:
            cls_path = alphabet_dir / cls
            files = sorted(list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png")))
            # Select up to 100 evenly spaced images for maximum diversity across the 3000 images
            step = max(1, len(files) // 100)
            selected_files = files[::step][:100]

            detected = 0
            t0 = time.time()
            for f in selected_files:
                if cls == "nothing":
                    row = dict(zip(feature_cols, [0.0] * TOTAL_SPATIAL_FEATURES))
                    row["label"] = cls
                    records.append(row)
                    detected += 1
                    continue

                bgr = cv2.imread(str(f))
                if bgr is None:
                    continue
                feat, _, _, _ = robust_extract_landmarks_from_bgr_frame(bgr)
                if feat is not None:
                    row = dict(zip(feature_cols, feat))
                    row["label"] = cls
                    records.append(row)
                    detected += 1

            dt = time.time() - t0
            print(f"  [+] Class '{cls}': {detected}/{len(selected_files)} extracted in {dt:.1f}s")
    else:
        print(f"[!] Warning: {alphabet_dir} not found.")

    # 2. Process 10 digit classes (0-9) from dataset/
    if digit_dir.exists():
        digit_classes = sorted([d.name for d in digit_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        print(f"[*] Processing {len(digit_classes)} digit classes from {digit_dir}...")
        for cls in digit_classes:
            cls_path = digit_dir / cls
            files = sorted(list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png")))
            detected = 0
            t0 = time.time()
            for f in files:
                bgr = cv2.imread(str(f))
                if bgr is None:
                    continue
                feat, _, _, _ = robust_extract_landmarks_from_bgr_frame(bgr)
                if feat is not None:
                    row = dict(zip(feature_cols, feat))
                    row["label"] = cls
                    records.append(row)
                    detected += 1
            dt = time.time() - t0
            print(f"  [+] Class '{cls}': {detected}/{len(files)} extracted in {dt:.1f}s")

    df = pd.DataFrame(records)
    print(f"\n[*] Total extracted records: {len(df)} across {df['label'].nunique()} classes.")
    df.to_csv(out_csv, index=False)
    print(f"[+] Saved dataset to {out_csv} ({out_csv.stat().st_size / 1024:.1f} KB).")

if __name__ == "__main__":
    main()
