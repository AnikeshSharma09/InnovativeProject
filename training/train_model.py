"""
Phase 2: Model Training.

Splits the cleaned dataset into:
- 80% Training Data
- 20% Testing Data (Stratified)
Trains a Random Forest classifier as the primary model,
with modular support for SVM / other classifiers for comparison.
Saves:
- models/sign_model.pkl
- models/label_encoder.pkl
- models/model_metadata.json
- data/X_test.csv & data/y_test.csv (held-out for Phase 3 evaluation)
"""
import os
import sys
import json
import argparse
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.landmarks_util import get_feature_names


def get_classifier(model_type: str = "rf", random_state: int = 42):
    """
    Returns an instantiated classifier based on the requested model type.
    Enables easy comparison between Random Forest, SVM, MLP, and KNN.
    """
    model_type = model_type.lower()
    if model_type in ["rf", "random_forest", "randomforest"]:
        print("[*] Instantiating Primary Classifier: Random Forest (n_estimators=100)")
        return RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type in ["svm", "svc", "support_vector"]:
        print("[*] Instantiating Alternative Classifier: Support Vector Machine (RBF kernel)")
        return SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            probability=True,  # Required for confidence probability estimation
            random_state=random_state
        )
    elif model_type in ["mlp", "neural_network"]:
        print("[*] Instantiating Multi-Layer Perceptron (MLP) Neural Network")
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=random_state
        )
    elif model_type in ["knn"]:
        print("[*] Instantiating K-Nearest Neighbors (KNN)")
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose from 'rf', 'svm', 'mlp', 'knn'.")


def train_sign_model(
    cleaned_csv: str = "data/landmarks_cleaned.csv",
    model_output_dir: str = "models",
    test_split_ratio: float = 0.20,
    model_type: str = "rf",
    random_state: int = 42
) -> None:
    """
    Trains the sign recognition model with stratified 80:20 split and saves artifacts.
    """
    cleaned_path = Path(cleaned_csv)
    models_dir = Path(model_output_dir)
    data_dir = cleaned_path.parent

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset '{cleaned_csv}' not found. "
            "Run 'python training/clean_data.py' first."
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("              PHASE 2: MODEL TRAINING                  ")
    print("=======================================================")

    df = pd.read_csv(cleaned_path)
    feature_cols = get_feature_names()

    X = df[feature_cols].values
    y_raw = df["label"].astype(str).values

    print(f"[*] Total dataset size: {len(X)} samples, {X.shape[1]} features")

    # Encode string labels into numerical indices
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = list(label_encoder.classes_)
    print(f"[*] Distinct ASL classes ({len(classes)}): {classes}")

    # Stratified 80:20 Split
    print(f"\n[*] Performing Stratified Train-Test Split (Train: {1.0 - test_split_ratio:.0%}, Test: {test_split_ratio:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_split_ratio,
        stratify=y,
        random_state=random_state
    )

    print(f"    - Training set size : {len(X_train)} samples")
    print(f"    - Testing set size  : {len(X_test)} samples (Held-out for Phase 3 evaluation)")

    # Save held-out test split for Phase 3 evaluation
    test_df_X = pd.DataFrame(X_test, columns=feature_cols)
    test_df_y = pd.DataFrame({
        "label_encoded": y_test,
        "label": label_encoder.inverse_transform(y_test)
    })
    test_df_X.to_csv(data_dir / "X_test.csv", index=False)
    test_df_y.to_csv(data_dir / "y_test.csv", index=False)
    print(f"[+] Saved held-out test data to: {data_dir / 'X_test.csv'} and {data_dir / 'y_test.csv'}")

    # Instantiate and Train Model
    clf = get_classifier(model_type, random_state=random_state)
    start_time = datetime.datetime.now()
    print(f"\n[*] Training {model_type.upper()} classifier...")
    clf.fit(X_train, y_train)
    training_duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"[+] Training completed in {training_duration:.2f} seconds.")

    # Preliminary training accuracy
    train_acc = clf.score(X_train, y_train)
    print(f"[*] Training Set Accuracy: {train_acc:.4f} ({train_acc:.2%})")

    # Save model and label encoder
    model_path = models_dir / "sign_model.pkl"
    encoder_path = models_dir / "label_encoder.pkl"
    joblib.dump(clf, model_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"[+] Saved trained model to: {model_path}")
    print(f"[+] Saved label encoder to: {encoder_path}")

    # Save metadata JSON
    metadata = {
        "model_type": model_type,
        "classifier_class": clf.__class__.__name__,
        "classes": classes,
        "num_classes": len(classes),
        "num_features": len(feature_cols),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_accuracy": float(train_acc),
        "training_duration_seconds": float(training_duration),
        "created_at": datetime.datetime.now().isoformat(),
        "random_state": random_state
    }
    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved model metadata to: {models_dir / 'model_metadata.json'}")

    print("\n[+] Phase 2 Model Training Complete!")
    print("    Next step: Run 'python training/evaluate_model.py' to evaluate on the unseen 20% test set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL Sign Recognition Classifier.")
    parser.add_argument("--cleaned-csv", type=str, default="data/landmarks_cleaned.csv", help="Cleaned dataset path")
    parser.add_argument("--model-dir", type=str, default="models", help="Destination folder for model artifacts")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "svm", "mlp", "knn"], help="Model architecture")
    parser.add_argument("--test-split", type=float, default=0.20, help="Test split ratio (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    train_sign_model(
        cleaned_csv=args.cleaned_csv,
        model_output_dir=args.model_dir,
        test_split_ratio=args.test_split,
        model_type=args.model,
        random_state=args.seed
    )
