"""
WLASL Word-Level Sign Language Feature & Sequence Extractor.

Processes real MP4 video clips from the WLASL dataset in 'archive/wlasl/videos/':
1. Targets high-frequency vocabulary words (e.g. help, drink, computer, before, go, yes, no, what, work, play, etc.).
2. Extracts 20-frame normalized MediaPipe landmark sequences (T=20, F=129).
3. Computes spatio-temporal summary features for instant high-speed classification.
4. Exports:
   - data/wlasl_X_seq.npy (N, 20, 129)
   - data/wlasl_X_feat.npy (N, 258)
   - data/wlasl_y.npy (N,)
   - models/wlasl_classes.json
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.landmarks_util import (
    normalize_landmarks,
    extract_motion_features,
    TOTAL_MOTION_FEATURES,
    DEFAULT_SEQUENCE_LENGTH,
    NUM_LANDMARKS
)


# Target clean vocabulary of high-frequency WLASL words with >= 10 available videos
TARGET_WORDS = [
    "before", "drink", "go", "computer", "who", "cousin", "help",
    "candy", "bed", "accident", "short", "yes", "no", "what",
    "later", "man", "shirt", "change", "call", "cold", "thin",
    "cool", "tall", "walk", "play", "work"
]


def extract_video_landmarks(
    video_path: Path,
    hands_detector,
    frame_start: int = 1,
    frame_end: int = -1,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts a normalized (20, 129) sequence and (258,) summary feature vector from a video.
    """
    if not video_path.exists() or video_path.stat().st_size < 1000:
        return None

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 3:
        cap.release()
        return None

    start = max(0, frame_start - 1)
    end = total_frames - 1
    if frame_end > 0 and frame_end > start:
        end = min(total_frames - 1, frame_end)

    if end <= start:
        start = 0
        end = total_frames - 1

    sample_indices = np.linspace(start, end, sequence_length).astype(int)
    target_set = set(sample_indices)

    frames_landmarks: List[Optional[List[Tuple[float, float, float]]]] = []
    curr_frame = 0
    sampled_dict = {}

    while cap.isOpened() and curr_frame <= end:
        ret, frame = cap.read()
        if not ret:
            break
        if curr_frame in target_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_detector.process(rgb)
            if res.multi_hand_landmarks:
                # Use primary hand (first detected)
                primary = res.multi_hand_landmarks[0]
                pts = [(lm.x, lm.y, lm.z) for lm in primary.landmark]
                sampled_dict[curr_frame] = pts
            else:
                sampled_dict[curr_frame] = None
        curr_frame += 1

    cap.release()

    # Reconstruct in order of sample_indices
    last_valid = None
    ordered_coords: List[List[Tuple[float, float, float]]] = []
    valid_count = 0

    for idx in sample_indices:
        pts = sampled_dict.get(idx)
        if pts is not None:
            last_valid = pts
            valid_count += 1
            ordered_coords.append(pts)
        elif last_valid is not None:
            ordered_coords.append(last_valid)
        else:
            ordered_coords.append(None)

    # If first frames were None, backfill from first valid
    if valid_count < 4:
        # Not enough hand detections in this video
        return None

    first_valid = next(p for p in ordered_coords if p is not None)
    for i in range(len(ordered_coords)):
        if ordered_coords[i] is None:
            ordered_coords[i] = first_valid

    # Compute 129-dim motion feature vectors across sequence
    seq_vectors: List[List[float]] = []
    prev_pts = None

    for pts in ordered_coords:
        m_vec, _ = extract_motion_features(pts, prev_pts)
        seq_vectors.append(m_vec)
        prev_pts = pts

    seq_array = np.array(seq_vectors, dtype=np.float32)  # (20, 129)

    # Compute spatio-temporal summary vector:
    # Initial shape (63) + Middle shape (63) + Final shape (63) + Mean velocity (63) + Max velocity (6) = 258 dims
    init_shape = seq_array[0, :63]
    mid_shape = seq_array[sequence_length // 2, :63]
    final_shape = seq_array[-1, :63]
    mean_vel = np.mean(seq_array[:, 63:126], axis=0)
    traj_delta = seq_array[-1, 126:129] - seq_array[0, 126:129]
    max_vel = np.max(np.abs(seq_array[:, 63:66]), axis=0)

    summary_feat = np.hstack([init_shape, mid_shape, final_shape, mean_vel, traj_delta, max_vel]).astype(np.float32)

    return seq_array, summary_feat


def extract_wlasl_dataset(
    json_path: str = "archive/wlasl/WLASL_v0.3.json",
    video_dir: str = "archive/wlasl/videos",
    output_dir: str = "data",
    max_per_word: int = 15
):
    print("=" * 65)
    print("      WLASL WORD-LEVEL SIGN LANGUAGE DATASET EXTRACTOR       ")
    print("=" * 65)

    v_dir = Path(video_dir)
    j_file = Path(json_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not j_file.exists():
        raise FileNotFoundError(f"WLASL JSON file not found: {j_file}")

    with open(j_file, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)

    # Build word -> instances lookup
    word_map = {}
    for item in wlasl_data:
        g = item["gloss"]
        if g in TARGET_WORDS:
            word_map[g] = item["instances"]

    print(f"[*] Target Vocabulary Words ({len(TARGET_WORDS)}): {TARGET_WORDS[:10]}...")

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.30,
        min_tracking_confidence=0.30
    )

    all_sequences = []
    all_summaries = []
    all_labels = []

    t0 = time.time()
    total_processed = 0

    for word in TARGET_WORDS:
        instances = word_map.get(word, [])
        word_success = 0
        print(f"\n[*] Processing Word: '{word}' (Available instances: {len(instances)})...")

        for inst in instances:
            if word_success >= max_per_word:
                break

            vid_id = inst["video_id"]
            v_path = v_dir / f"{vid_id}.mp4"
            if not v_path.exists():
                continue

            f_start = inst.get("frame_start", 1)
            f_end = inst.get("frame_end", -1)

            res = extract_video_landmarks(v_path, mp_hands, f_start, f_end)
            if res is not None:
                seq, summary = res
                all_sequences.append(seq)
                all_summaries.append(summary)
                all_labels.append(word)
                word_success += 1
                total_processed += 1

        print(f"    [+] '{word}': Successfully extracted {word_success} video sequences.")

    mp_hands.close()
    duration = time.time() - t0
    print(f"\n[+] Extraction Complete in {duration:.1f}s!")
    print(f"    Total Extracted Instances : {len(all_labels)}")
    print(f"    Unique Classes           : {len(set(all_labels))}")

    # Convert to NumPy
    X_seq = np.array(all_sequences, dtype=np.float32)
    X_feat = np.array(all_summaries, dtype=np.float32)
    y = np.array(all_labels)

    # Save to data/
    np.save(str(out_dir / "wlasl_X_seq.npy"), X_seq)
    np.save(str(out_dir / "wlasl_X_feat.npy"), X_feat)
    np.save(str(out_dir / "wlasl_y.npy"), y)

    # Save classes JSON
    classes_list = sorted(list(set(all_labels)))
    with open("models/wlasl_classes.json", "w", encoding="utf-8") as f:
        json.dump(classes_list, f, indent=4)

    print(f"[+] Saved wlasl_X_seq.npy with shape: {X_seq.shape}")
    print(f"[+] Saved wlasl_X_feat.npy with shape: {X_feat.shape}")
    print(f"[+] Saved wlasl_y.npy with shape: {y.shape}")
    print(f"[+] Saved models/wlasl_classes.json ({len(classes_list)} classes)")


if __name__ == "__main__":
    extract_wlasl_dataset()
