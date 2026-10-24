"""
Training Pipeline for WLASL Word-Level Sign Language Model.

Trains an Ensemble Classifier (ExtraTrees + RandomForest + HistGradientBoosting)
on spatio-temporal sign language features extracted from the WLASL video dataset.
Saves:
- models/wlasl_word_model.pkl
- models/wlasl_label_encoder.pkl
- models/wlasl_metadata.json
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_wlasl_model(
    feat_path: str = "data/wlasl_X_feat.npy",
    y_path: str = "data/wlasl_y.npy",
    model_output_dir: str = "models",
    random_state: int = 42
):
    print("=" * 65)
    print("      TRAINING WLASL WORD-LEVEL SIGN LANGUAGE MODEL          ")
    print("=" * 65)

    f_file = Path(feat_path)
    y_file = Path(y_path)
    out_dir = Path(model_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not f_file.exists() or not y_file.exists():
        raise FileNotFoundError(f"Extracted feature files not found: {f_file}, {y_file}")

    X = np.load(str(f_file))
    y_raw = np.load(str(y_file))

    print(f"[*] Total dataset samples : {len(X)} instances")
    print(f"[*] Feature dimensions    : {X.shape[1]} features")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_raw)
    classes = list(encoder.classes_)
    print(f"[*] Distinct word classes : {len(classes)} words: {classes[:10]}...")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=random_state, stratify=y_enc
    )
    print(f"[*] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Multi-Model Ensemble
    print("[*] Training High-Accuracy Ensemble (ExtraTrees + RandomForest + HistGB)...")
    t0 = time.time()

    et = ExtraTreesClassifier(n_estimators=180, max_depth=22, min_samples_leaf=1, random_state=42, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=160, max_depth=22, min_samples_leaf=1, random_state=42, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42)

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

    ensemble.fit(X_train, y_train)
    train_duration = time.time() - t0
    print(f"[+] Ensemble trained in {train_duration:.2f} seconds.")

    # Evaluate on held-out test split
    test_preds = ensemble.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print("\n" + "=" * 65)
    print(f"  WLASL HELD-OUT TEST ACCURACY: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print("=" * 65)

    y_test_names = encoder.inverse_transform(y_test)
    pred_names = encoder.inverse_transform(test_preds)
    report = classification_report(y_test_names, pred_names, output_dict=True)
    print(f"  Macro Precision : {report['macro avg']['precision']:.4f}")
    print(f"  Macro Recall    : {report['macro avg']['recall']:.4f}")
    print(f"  Macro F1-Score  : {report['macro avg']['f1-score']:.4f}")

    # Save models
    model_path = out_dir / "wlasl_word_model.pkl"
    encoder_path = out_dir / "wlasl_label_encoder.pkl"
    meta_path = out_dir / "wlasl_metadata.json"

    joblib.dump(ensemble, str(model_path), compress=3)
    joblib.dump(encoder, str(encoder_path))

    meta = {
        "model_type": "WLASL_Word_Ensemble",
        "classes": classes,
        "num_classes": len(classes),
        "feature_dim": int(X.shape[1]),
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_accuracy": float(test_acc),
        "macro_f1": float(report['macro avg']['f1-score']),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print(f"[+] Saved model to: {model_path}")
    print(f"[+] Saved encoder to: {encoder_path}")
    print(f"[+] Saved metadata to: {meta_path}")


if __name__ == "__main__":
    train_wlasl_model()
