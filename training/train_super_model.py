"""
High-Precision Sign Language Classifier Trainer.
Features:
1. Synthetic 3D Landmark Augmentation (Rotation, Scaling, Gaussian Jitter).
2. 20 Physical Geometric Invariants (Fingertip distances, curl ratios, spread angles).
3. Optimized Ensemble Classifier (ExtraTrees + RandomForest + HistGradientBoosting).
4. Generates updated 'models/sign_model.pkl' and evaluation benchmarks.
"""
import os
import sys
import json
import time
from typing import Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.enhanced_classifier import EnhancedSignClassifier, compute_geometric_invariants


def augment_landmarks(X_batch: np.ndarray, y_batch: np.ndarray, num_copies: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies realistic 3D landmark rotations (+-12 deg), scaling (0.92-1.08x),
    and subtle gaussian jitter to make the classifier resilient to camera angles.
    """
    aug_X_list = [X_batch]
    aug_y_list = [y_batch]
    N = X_batch.shape[0]
    pts = X_batch.reshape(N, 21, 3)

    for _ in range(num_copies):
        # 1. Random 2D in-plane rotation angle
        angles = np.random.uniform(-0.21, 0.21, size=N)  # ~12 degrees
        cos_a = np.cos(angles)[:, None, None]
        sin_a = np.sin(angles)[:, None, None]

        px = pts[:, :, 0:1]
        py = pts[:, :, 1:2]
        pz = pts[:, :, 2:3]

        rot_x = px * cos_a - py * sin_a
        rot_y = px * sin_a + py * cos_a

        # 2. Scale jitter
        scales = np.random.uniform(0.93, 1.07, size=(N, 1, 1))

        # 3. Micro gaussian sensor noise
        noise = np.random.normal(0, 0.0035, size=(N, 21, 3))

        aug_pts = np.concatenate([rot_x, rot_y, pz], axis=-1) * scales + noise
        aug_X_flat = aug_pts.reshape(N, 63)

        aug_X_list.append(aug_X_flat)
        aug_y_list.append(y_batch)

    return np.vstack(aug_X_list), np.concatenate(aug_y_list)


def main():
    print("=" * 65)
    print("  HIGH-PRECISION SIGN LANGUAGE CLASSIFIER TRAINING PIPELINE  ")
    print("=" * 65)

    data_path = Path("data/landmarks_cleaned.csv")
    if not data_path.exists():
        data_path = Path("data/landmarks.csv")
    if not data_path.exists():
        print("[!] Error: No landmark dataset found in data/.")
        return

    print(f"[*] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "label"]
    raw_X = np.array(df[feature_cols].values, dtype=np.float32)
    raw_y = np.array(df["label"].astype(str).values)

    encoder = joblib.load("models/label_encoder.pkl")
    classes = list(encoder.classes_)
    y_encoded = encoder.transform(raw_y)

    print(f"[*] Total dataset samples: {len(raw_X)} across {len(classes)} classes.")

    # 80/20 Stratified Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw_X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    print(f"[*] Base training samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")

    # Apply 3D Synthetic Landmark Augmentation (3x expansion)
    print("[*] Generating 3D synthetic landmark augmentations (Rotation, Scale, Noise)...")
    np.random.seed(42)
    X_train_aug_raw, y_train_aug = augment_landmarks(X_train_raw, y_train, num_copies=3)
    print(f"[+] Training dataset expanded: {len(X_train_raw)} -> {len(X_train_aug_raw)} samples.")

    # Compute 20 Physical Geometric Invariant Features
    print("[*] Computing physical geometric invariant features (63 -> 83 features)...")
    X_train_feat = compute_geometric_invariants(X_train_aug_raw)
    X_test_feat = compute_geometric_invariants(X_test_raw)

    print(f"[+] Feature matrix prepared with shape: {X_train_feat.shape}")

    # Build Ensemble
    print("[*] Training High-Accuracy Ensemble (ExtraTrees + RandomForest + HistGradientBoosting)...")
    t0 = time.time()
    
    et = ExtraTreesClassifier(n_estimators=160, max_depth=24, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=140, max_depth=24, min_samples_leaf=2, random_state=42, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.08, random_state=42)

    ensemble = VotingClassifier(
        estimators=[
            ("et", et),
            ("rf", rf),
            ("hgb", hgb)
        ],
        voting="soft",
        weights=[3, 2, 2],
        n_jobs=-1
    )

    ensemble.fit(X_train_feat, y_train_aug)
    train_time = time.time() - t0
    print(f"[+] Ensemble trained in {train_time:.2f} seconds.")

    # Evaluate on Raw Held-Out Test Set
    test_preds = ensemble.predict(X_test_feat)
    test_acc = accuracy_score(y_test, test_preds)
    print("\n" + "=" * 65)
    print(f"  HELD-OUT TEST SET ACCURACY: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print("=" * 65)

    y_test_names = encoder.inverse_transform(y_test)
    pred_names = encoder.inverse_transform(test_preds)

    rep = classification_report(y_test_names, pred_names, output_dict=True)
    print(f"  Macro Precision : {rep['macro avg']['precision']:.4f}")
    print(f"  Macro Recall    : {rep['macro avg']['recall']:.4f}")
    print(f"  Macro F1-Score  : {rep['macro avg']['f1-score']:.4f}")

    # Package into backward-compatible EnhancedSignClassifier wrapper
    wrapped_model = EnhancedSignClassifier(ensemble, classes=np.arange(len(classes)))

    # Save to models/sign_model.pkl (compressed to <20MB for GitHub limits)
    out_path = Path("models/sign_model.pkl")
    joblib.dump(wrapped_model, out_path, compress=3)
    print(f"\n[+] Saved upgraded ensemble model to {out_path.resolve()}")

    # Save updated metadata
    meta = {
        "model_type": "ensemble_stack",
        "classifier_class": "VotingClassifier(ExtraTrees+RF+HistGB)",
        "classes": classes,
        "num_classes": len(classes),
        "num_features": 83,
        "base_features": 63,
        "geometric_features": 20,
        "train_samples": int(len(X_train_feat)),
        "test_samples": int(len(X_test_feat)),
        "test_accuracy": float(test_acc),
        "training_duration_seconds": float(train_time),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    print("[+] Updated models/model_metadata.json")


if __name__ == "__main__":
    main()
