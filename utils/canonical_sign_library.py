"""
Canonical Sign Landmark Library
================================
Extracts, regularizes, and caches canonical 3D skeletal landmark coordinates (21 joints x [x, y, z])
for ASL fingerspelling (A-Z, 0-9, space, rest) and whole-word signs.
Provides instant lookup (<1ms) for real-time 3D animation and reverse speech-to-sign avatar synthesis.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CANONICAL_CACHE_PATH = DATA_DIR / "canonical_signs_3d.json"

# MediaPipe Hand Skeletal Connections
HAND_CONNECTIONS = [
    # Palm Base / Carpal
    [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
    # Thumb
    [1, 2], [2, 3], [3, 4],
    # Index
    [5, 6], [6, 7], [7, 8],
    # Middle
    [9, 10], [10, 11], [11, 12],
    # Ring
    [13, 14], [14, 15], [15, 16],
    # Pinky
    [17, 18], [18, 19], [19, 20]
]

# Finger groupings for distinct color rendering
FINGER_GROUPS = {
    "wrist_palm": [0, 1, 5, 9, 13, 17],
    "thumb": [2, 3, 4],
    "index": [6, 7, 8],
    "middle": [10, 11, 12],
    "ring": [14, 15, 16],
    "pinky": [18, 19, 20]
}


def _create_conversational_word_landmarks() -> Dict[str, List[List[float]]]:
    """
    Creates foundational 3D postures for high-frequency conversational words:
    'THANK', 'YOU', 'I', 'YES', 'NO', 'PLEASE'.
    """
    words = {}

    # 1. 'YOU': Pointing index finger forward into camera/viewer
    words["YOU"] = [
        [0.0, 0.0, 0.0],          # 0: Wrist
        [0.10, -0.12, -0.05],     # 1: Thumb CMC
        [0.15, -0.20, -0.06],     # 2: Thumb MCP
        [0.12, -0.25, -0.06],     # 3: Thumb IP
        [0.08, -0.28, -0.06],     # 4: Thumb Tip (tucked)
        [0.06, -0.32, -0.05],     # 5: Index MCP
        [0.06, -0.45, -0.25],     # 6: Index PIP (pointing forward)
        [0.06, -0.55, -0.45],     # 7: Index DIP
        [0.05, -0.62, -0.65],     # 8: Index Tip (extended straight forward toward viewer)
        [-0.04, -0.30, -0.04],    # 9: Middle MCP
        [-0.05, -0.22, -0.15],    # 10: Middle PIP (curled)
        [-0.05, -0.15, -0.12],    # 11: Middle DIP
        [-0.05, -0.10, -0.08],    # 12: Middle Tip
        [-0.12, -0.28, -0.03],    # 13: Ring MCP
        [-0.13, -0.20, -0.14],    # 14: Ring PIP (curled)
        [-0.13, -0.14, -0.11],    # 15: Ring DIP
        [-0.13, -0.09, -0.07],    # 16: Ring Tip
        [-0.19, -0.24, -0.02],    # 17: Pinky MCP
        [-0.20, -0.18, -0.12],    # 18: Pinky PIP (curled)
        [-0.20, -0.12, -0.09],    # 19: Pinky DIP
        [-0.20, -0.08, -0.06]     # 20: Pinky Tip
    ]

    # 2. 'THANK' / 'THANKS': Open flat hand extended forward with palm up/in
    words["THANK"] = [
        [0.0, 0.0, 0.0],
        [0.20, -0.14, -0.08], [0.28, -0.25, -0.12], [0.32, -0.35, -0.15], [0.35, -0.44, -0.18],
        [0.12, -0.38, -0.15], [0.14, -0.55, -0.25], [0.15, -0.70, -0.32], [0.15, -0.82, -0.38],
        [0.02, -0.40, -0.15], [0.03, -0.58, -0.25], [0.03, -0.73, -0.32], [0.03, -0.85, -0.38],
        [-0.08, -0.38, -0.14], [-0.08, -0.55, -0.24], [-0.08, -0.69, -0.31], [-0.08, -0.81, -0.37],
        [-0.17, -0.34, -0.13], [-0.18, -0.48, -0.22], [-0.18, -0.60, -0.28], [-0.18, -0.70, -0.33]
    ]
    words["THANKS"] = words["THANK"]

    # 3. 'YES': S-fist tilted forward
    words["YES"] = [
        [0.0, 0.0, 0.0],
        [0.08, -0.12, -0.08], [0.12, -0.20, -0.12], [0.08, -0.26, -0.14], [0.00, -0.28, -0.14],
        [0.06, -0.30, -0.06], [0.06, -0.20, -0.18], [0.04, -0.14, -0.15], [0.02, -0.10, -0.10],
        [-0.02, -0.30, -0.05], [-0.02, -0.20, -0.18], [-0.03, -0.14, -0.15], [-0.04, -0.10, -0.10],
        [-0.09, -0.28, -0.04], [-0.09, -0.19, -0.17], [-0.09, -0.13, -0.14], [-0.09, -0.09, -0.09],
        [-0.16, -0.24, -0.03], [-0.16, -0.17, -0.15], [-0.16, -0.12, -0.13], [-0.16, -0.08, -0.08]
    ]

    # 4. 'NO': Index and middle snapping onto thumb tip
    words["NO"] = [
        [0.0, 0.0, 0.0],
        [0.10, -0.14, -0.06], [0.15, -0.24, -0.09], [0.14, -0.33, -0.12], [0.08, -0.38, -0.14],
        [0.08, -0.32, -0.05], [0.08, -0.42, -0.14], [0.08, -0.44, -0.18], [0.07, -0.40, -0.15],
        [-0.01, -0.32, -0.05], [-0.01, -0.42, -0.14], [-0.01, -0.44, -0.18], [-0.02, -0.40, -0.15],
        [-0.10, -0.28, -0.04], [-0.11, -0.20, -0.15], [-0.11, -0.14, -0.12], [-0.11, -0.09, -0.08],
        [-0.18, -0.24, -0.03], [-0.19, -0.18, -0.13], [-0.19, -0.12, -0.10], [-0.19, -0.08, -0.07]
    ]

    # 5. 'PLEASE': Open flat hand with thumb extended
    words["PLEASE"] = words["THANK"]

    return words


def _create_neutral_rest_landmarks() -> List[List[float]]:
    """
    Generates a natural, anatomically relaxed 3D hand resting posture.
    Used for smooth transitions between words and sentences.
    """
    # 21 points relative to wrist at (0, 0, 0)
    # Fingers gently curved forward and inward
    rest = [
        [0.0, 0.0, 0.0],          # 0: Wrist
        [0.18, -0.12, -0.05],     # 1: Thumb CMC
        [0.26, -0.22, -0.08],     # 2: Thumb MCP
        [0.28, -0.32, -0.10],     # 3: Thumb IP
        [0.26, -0.40, -0.12],     # 4: Thumb Tip (relaxed, curled in)
        [0.14, -0.40, -0.04],     # 5: Index MCP
        [0.15, -0.56, -0.10],     # 6: Index PIP
        [0.14, -0.68, -0.16],     # 7: Index DIP
        [0.12, -0.74, -0.22],     # 8: Index Tip (gentle natural curve)
        [0.02, -0.42, -0.03],     # 9: Middle MCP
        [0.02, -0.58, -0.11],     # 10: Middle PIP
        [0.01, -0.71, -0.18],     # 11: Middle DIP
        [0.00, -0.78, -0.24],     # 12: Middle Tip
        [-0.09, -0.39, -0.04],    # 13: Ring MCP
        [-0.10, -0.54, -0.11],    # 14: Ring PIP
        [-0.10, -0.65, -0.17],    # 15: Ring DIP
        [-0.10, -0.72, -0.22],    # 16: Ring Tip
        [-0.19, -0.34, -0.05],    # 17: Pinky MCP
        [-0.21, -0.47, -0.11],    # 18: Pinky PIP
        [-0.21, -0.56, -0.16],    # 19: Pinky DIP
        [-0.20, -0.62, -0.20]     # 20: Pinky Tip
    ]
    return rest


def build_canonical_library() -> Dict[str, List[List[float]]]:
    """
    Computes robust median 3D landmarks for all static classes from dataset
    and integrates whole-word sign postures.
    """
    canonical: Dict[str, List[List[float]]] = {}

    # 1. Base Alphabet (A-Z) and Digits (0-9) from landmarks_combined.csv
    csv_path = DATA_DIR / "landmarks_combined.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            coord_cols = [c for c in df.columns if c != "label"]
            for lbl, group in df.groupby("label"):
                lbl_str = str(lbl).strip().upper()
                # Compute median per coordinate to eliminate outliers/jitter
                med_coords = group[coord_cols].median().values
                pts_21 = med_coords.reshape(21, 3).tolist()
                # Round to 4 decimal places for clean serialization
                canonical[lbl_str] = [[round(v, 4) for v in pt] for pt in pts_21]
        except Exception as e:
            print(f"[!] Warning: Could not compute canonical landmarks from CSV: {e}")

    # 2. Extract WLASL Words from features if available
    wlasl_feat_path = DATA_DIR / "wlasl_X_feat.npy"
    wlasl_y_path = DATA_DIR / "wlasl_y.npy"
    if wlasl_feat_path.exists() and wlasl_y_path.exists():
        try:
            feats = np.load(wlasl_feat_path)
            labels = np.load(wlasl_y_path)
            for w in np.unique(labels):
                w_str = str(w).strip().upper()
                idx = np.where(labels == w)[0]
                if len(idx) > 0:
                    med_w = np.median(feats[idx, :63], axis=0)
                    pts_21 = med_w.reshape(21, 3).tolist()
                    canonical[w_str] = [[round(v, 4) for v in pt] for pt in pts_21]
        except Exception as e:
            print(f"[!] Warning: Could not compute WLASL word canonical landmarks: {e}")

    # 3. Add Conversational Core Words (THANK, YOU, YES, NO, PLEASE)
    for c_word, c_pts in _create_conversational_word_landmarks().items():
        canonical[c_word] = c_pts

    # 4. Add Resting and Control Postures
    canonical["REST"] = _create_neutral_rest_landmarks()
    if "SPACE" not in canonical and "space" in canonical:
        canonical["SPACE"] = canonical["space"]
    elif "SPACE" not in canonical:
        # Flat horizontal hand
        canonical["SPACE"] = [
            [0.0, 0.0, 0.0],
            [0.15, -0.10, -0.05], [0.22, -0.18, -0.08], [0.26, -0.25, -0.10], [0.28, -0.32, -0.12],
            [-0.30, -0.35, -0.05], [-0.45, -0.38, -0.08], [-0.58, -0.40, -0.10], [-0.68, -0.42, -0.12],
            [-0.32, -0.40, -0.05], [-0.48, -0.44, -0.08], [-0.62, -0.46, -0.10], [-0.72, -0.48, -0.12],
            [-0.33, -0.44, -0.05], [-0.49, -0.48, -0.08], [-0.62, -0.50, -0.10], [-0.71, -0.52, -0.12],
            [-0.32, -0.47, -0.05], [-0.46, -0.50, -0.08], [-0.57, -0.52, -0.10], [-0.65, -0.54, -0.12]
        ]

    # Save to disk cache
    try:
        with open(CANONICAL_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(canonical, f, indent=2)
        print(f"[+] Canonical Sign Library saved: {len(canonical)} signs -> {CANONICAL_CACHE_PATH}")
    except Exception as e:
        print(f"[!] Warning: Could not save canonical cache: {e}")

    return canonical


def get_canonical_library() -> Dict[str, List[List[float]]]:
    """Loads canonical sign library from cache, building it if missing."""
    if CANONICAL_CACHE_PATH.exists():
        try:
            with open(CANONICAL_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if len(data) >= 30:
                    return data
        except Exception:
            pass
    return build_canonical_library()


if __name__ == "__main__":
    lib = build_canonical_library()
    print(f"Verified {len(lib)} canonical signs in library.")
    print("Available keys sample:", list(lib.keys())[:15])
