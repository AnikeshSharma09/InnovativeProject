"""
Phase 3: Model Evaluation.

Evaluates the trained classifier strictly on the unseen 20% test dataset:
- Calculates true Accuracy, Precision, Recall, and F1-score (macro and weighted).
- Generates a full Scikit-Learn classification report.
- Plots and saves the complete Confusion Matrix via Seaborn/Matplotlib.
- Exports structured results to models/evaluation_results.json and models/evaluation_report.txt.
- NO hard-coded or fake metrics. All values are calculated from the actual model.
"""
import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import matplotlib
# Use Agg backend for non-interactive server/script environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def evaluate_trained_model(
    model_path: str = "models/sign_model.pkl",
    encoder_path: str = "models/label_encoder.pkl",
    test_features_csv: str = "data/X_test.csv",
    test_labels_csv: str = "data/y_test.csv",
    output_dir: str = "models"
) -> dict:
    """
    Loads trained model and tests on held-out 20% test partition.
    """
    model_file = Path(model_path)
    encoder_file = Path(encoder_path)
    x_test_file = Path(test_features_csv)
    y_test_file = Path(test_labels_csv)
    out_dir = Path(output_dir)

    for p, name in [
        (model_file, "Trained Model"),
        (encoder_file, "Label Encoder"),
        (x_test_file, "X_test Features"),
        (y_test_file, "y_test Labels")
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"{name} file not found at '{p}'. "
                "Ensure you ran 'python training/train_model.py' first."
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("             PHASE 3: MODEL EVALUATION                 ")
    print("=======================================================")

    # 1. Load Model, Label Encoder, and Test Data
    clf = joblib.load(model_file)
    label_encoder = joblib.load(encoder_file)
    X_test_df = pd.read_csv(x_test_file)
    y_test_df = pd.read_csv(y_test_file)

    X_test = X_test_df.values
    y_test = y_test_df["label_encoded"].values
    class_names = [str(c) for c in label_encoder.classes_]

    print(f"[*] Loaded trained model: {clf.__class__.__name__}")
    print(f"[*] Total unseen test samples: {len(X_test)}")
    print(f"[*] Classes evaluated ({len(class_names)}): {class_names}")

    # 2. Run Predictions on Test Set
    print("\n[*] Generating predictions on unseen test dataset...")
    y_pred = clf.predict(X_test)

    # 3. Calculate Core Performance Metrics (Strictly derived from model predictions)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # 4. Generate Classification Report
    clf_report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    clf_report_text = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("              CLASSIFICATION PERFORMANCE REPORT         ")
    print("=" * 60)
    print(f"Accuracy           : {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision (Macro)  : {precision_macro:.4f} ({precision_macro:.2%})")
    print(f"Precision (Weight) : {precision_weighted:.4f} ({precision_weighted:.2%})")
    print(f"Recall (Macro)     : {recall_macro:.4f} ({recall_macro:.2%})")
    print(f"Recall (Weight)    : {recall_weighted:.4f} ({recall_weighted:.2%})")
    print(f"F1-Score (Macro)   : {f1_macro:.4f} ({f1_macro:.2%})")
    print(f"F1-Score (Weight)  : {f1_weighted:.4f} ({f1_weighted:.2%})")
    print("-" * 60)
    print(clf_report_text)
    print("=" * 60)

    # 5. Generate and Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    cm_plot_path = out_dir / "confusion_matrix.png"

    # Scale figure size dynamically based on number of classes
    fig_dim = max(10, int(len(class_names) * 0.45))
    plt.figure(figsize=(fig_dim, fig_dim - 2))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title(f"ASL Recognition Confusion Matrix\n({clf.__class__.__name__} - Test Accuracy: {accuracy:.2%})", fontsize=14, pad=15)
    plt.xlabel("Predicted ASL Class", fontsize=12, labelpad=10)
    plt.ylabel("True Ground Truth Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix visualization to: {cm_plot_path}")

    # 6. Save Structured Evaluation Results
    evaluation_summary = {
        "model_architecture": clf.__class__.__name__,
        "test_samples_count": len(X_test),
        "num_classes": len(class_names),
        "classes": class_names,
        "metrics": {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "f1_score_macro": f1_macro,
            "f1_score_weighted": f1_weighted
        },
        "per_class_report": clf_report_dict
    }

    json_report_path = out_dir / "evaluation_results.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=4)
    print(f"[+] Saved metrics JSON to: {json_report_path}")

    # Save human-readable text report
    txt_report_path = out_dir / "evaluation_report.txt"
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("=======================================================\n")
        f.write("      ASL MODEL EVALUATION REPORT (HELD-OUT 20% TEST)  \n")
        f.write("=======================================================\n\n")
        f.write(f"Model Architecture : {clf.__class__.__name__}\n")
        f.write(f"Total Test Samples : {len(X_test)}\n")
        f.write(f"Overall Accuracy   : {accuracy:.4f} ({accuracy:.2%})\n")
        f.write(f"Macro Precision    : {precision_macro:.4f}\n")
        f.write(f"Macro Recall       : {recall_macro:.4f}\n")
        f.write(f"Macro F1-Score     : {f1_macro:.4f}\n")
        f.write(f"Weighted F1-Score  : {f1_weighted:.4f}\n\n")
        f.write("Detailed Per-Class Performance:\n")
        f.write("-" * 60 + "\n")
        f.write(clf_report_text + "\n")
    print(f"[+] Saved text evaluation report to: {txt_report_path}")

    print("\n[+] Phase 3 Model Evaluation Complete!")
    return evaluation_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained ASL model on unseen test split.")
    parser.add_argument("--model-path", type=str, default="models/sign_model.pkl", help="Path to trained model")
    parser.add_argument("--encoder-path", type=str, default="models/label_encoder.pkl", help="Path to label encoder")
    parser.add_argument("--x-test", type=str, default="data/X_test.csv", help="Held-out X_test CSV path")
    parser.add_argument("--y-test", type=str, default="data/y_test.csv", help="Held-out y_test CSV path")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for reports")
    args = parser.parse_args()

    evaluate_trained_model(
        model_path=args.model_path,
        encoder_path=args.encoder_path,
        test_features_csv=args.x_test,
        test_labels_csv=args.y_test,
        output_dir=args.output_dir
    )
