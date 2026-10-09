"""
Phase 3: Deep Learning LSTM Model Evaluation.

Evaluates MotionSignLSTM strictly on held-out 20% test motion sequences:
- Calculates true Accuracy, Precision, Recall, and F1-score.
- Generates Scikit-Learn Classification Report.
- Plots and saves high-resolution Confusion Matrix via Seaborn to models/confusion_matrix.png.
- Exports metrics to models/evaluation_results.json and models/evaluation_report.txt.
"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
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

from models.lstm_model import load_lstm_checkpoint


def evaluate_lstm_model(
    model_path: str = "models/sign_lstm.pth",
    encoder_path: str = "models/label_encoder.pkl",
    test_x_path: str = "data/X_test_seq.npy",
    test_y_path: str = "data/y_test_seq.npy",
    output_dir: str = "models"
) -> dict:
    """Evaluates the trained LSTM on held-out test sequence partition."""
    m_file = Path(model_path)
    e_file = Path(encoder_path)
    tx_file = Path(test_x_path)
    ty_file = Path(test_y_path)
    out_dir = Path(output_dir)

    for p, name in [
        (m_file, "LSTM Checkpoint"),
        (e_file, "Label Encoder"),
        (tx_file, "Test Sequences X"),
        (ty_file, "Test Labels y")
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"{name} not found at: '{p}'. "
                "Ensure you ran 'python training/train_lstm.py' first."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=======================================================")
    print("        PHASE 3: MOTION-AWARE LSTM MODEL EVALUATION    ")
    print("=======================================================")

    # 1. Load Model & Label Encoder
    model, metadata = load_lstm_checkpoint(str(m_file), device=device)
    label_encoder = joblib.load(e_file)
    class_names = [str(c) for c in label_encoder.classes_]

    # 2. Load Held-Out Test Sequences
    X_test = np.load(str(tx_file))
    y_test = np.load(str(ty_file))

    print(f"[*] Loaded LSTM Checkpoint : {model.__class__.__name__}")
    print(f"[*] Unseen Test Sequences  : {len(X_test)} samples of shape {X_test.shape[1:]}")
    print(f"[*] Classes Evaluated ({len(class_names)}) : {class_names[:8]}...")

    # 3. Predict on Test Set
    x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    probs_tensor = model.predict_probabilities(x_tensor)
    y_pred = torch.argmax(probs_tensor, dim=1).cpu().numpy()

    # 4. Calculate Grounded Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    clf_report_dict = classification_report(
        y_test,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    clf_report_text = classification_report(
        y_test,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("          LSTM CLASSIFICATION PERFORMANCE REPORT        ")
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
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    cm_plot_path = out_dir / "confusion_matrix.png"

    fig_dim = max(10, int(len(class_names) * 0.45))
    plt.figure(figsize=(fig_dim, fig_dim - 2))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="crest",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title(f"Motion-Aware LSTM Confusion Matrix\n(Test Accuracy: {accuracy:.2%})", fontsize=14, pad=15)
    plt.xlabel("Predicted ASL Class", fontsize=12, labelpad=10)
    plt.ylabel("True Ground Truth Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix to: {cm_plot_path}")

    # 6. Save Evaluation Results JSON & TXT
    eval_results = {
        "model_architecture": "MotionSignLSTM (PyTorch)",
        "sequence_length": int(X_test.shape[1]),
        "features_per_frame": int(X_test.shape[2]),
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

    json_path = out_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print(f"[+] Saved metrics JSON to: {json_path}")

    txt_path = out_dir / "evaluation_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=======================================================\n")
        f.write("    MOTION-AWARE LSTM EVALUATION (HELD-OUT 20% TEST)   \n")
        f.write("=======================================================\n\n")
        f.write(f"Model Architecture : MotionSignLSTM (PyTorch)\n")
        f.write(f"Sequence Length    : {X_test.shape[1]} frames\n")
        f.write(f"Total Test Samples : {len(X_test)}\n")
        f.write(f"Overall Accuracy   : {accuracy:.4f} ({accuracy:.2%})\n")
        f.write(f"Macro Precision    : {precision_macro:.4f}\n")
        f.write(f"Macro Recall       : {recall_macro:.4f}\n")
        f.write(f"Macro F1-Score     : {f1_macro:.4f}\n")
        f.write(f"Weighted F1-Score  : {f1_weighted:.4f}\n\n")
        f.write("Detailed Per-Class Performance:\n")
        f.write("-" * 60 + "\n")
        f.write(clf_report_text + "\n")
    print(f"[+] Saved text evaluation report to: {txt_path}")

    print("\n[+] Phase 3 LSTM Evaluation Complete!")
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Motion-Aware LSTM on test sequences.")
    parser.add_argument("--model", type=str, default="models/sign_lstm.pth", help="Path to LSTM checkpoint")
    parser.add_argument("--encoder", type=str, default="models/label_encoder.pkl", help="Path to label encoder")
    parser.add_argument("--test-x", type=str, default="data/X_test_seq.npy", help="Test X sequences")
    parser.add_argument("--test-y", type=str, default="data/y_test_seq.npy", help="Test y labels")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    args = parser.parse_args()

    evaluate_lstm_model(
        model_path=args.model,
        encoder_path=args.encoder,
        test_x_path=args.test_x,
        test_y_path=args.test_y,
        output_dir=args.output_dir
    )
