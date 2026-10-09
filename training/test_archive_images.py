"""
Comprehensive Verification Script for Kaggle ASL Alphabet Test Images.
Evaluates all test images in archive/asl_alphabet_test/asl_alphabet_test/
using the robust cascading hand detector and SignPredictor.
Outputs a detailed verification table and performance metrics.
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction.predictor import SignPredictor


def evaluate_test_archive(
    test_dir: str = "archive/asl_alphabet_test/asl_alphabet_test",
    model_type: str = "rf"
):
    test_path = Path(test_dir)
    if not test_path.exists():
        # Check fallback
        alt_path = Path("archive/asl_alphabet_test")
        if alt_path.exists():
            test_path = alt_path
        else:
            print(f"[!] Error: Test directory not found at '{test_path}'")
            return

    image_files = sorted([f for f in test_path.glob("*.jpg")])
    if not image_files:
        print(f"[!] No JPG files found in '{test_path}'")
        return

    print("=" * 80)
    print("      KAGGLE ASL ALPHABET TEST DATASET VERIFICATION BENCHMARK      ")
    print("=" * 80)
    print(f"[*] Target Directory : {test_path.resolve()}")
    print(f"[*] Total Test Files : {len(image_files)}")
    print(f"[*] Inference Engine : {'Random Forest (99.33% Baseline)' if model_type == 'rf' else 'Deep LSTM'}")
    print("=" * 80)

    predictor = SignPredictor()

    results = []
    correct_count = 0
    total_hand_images = 0
    detected_hand_images = 0

    header = f"{'Filename':18} | {'Expected':8} | {'Detected':8} | {'Predicted':9} | {'Confidence':10} | {'Status':20} | {'Match'}"
    print(header)
    print("-" * len(header))

    for img_path in image_files:
        expected = img_path.stem.replace("_test", "")
        img = cv2.imread(str(img_path))

        out, _ = predictor.predict_image(img, model_type=model_type)

        is_hand_expected = (expected != "nothing")
        if is_hand_expected:
            total_hand_images += 1

        is_detected = out.hand_detected
        pred_label = out.predicted_label if is_detected else "NONE"
        conf_str = f"{out.confidence_score:.1f}%" if is_detected else "N/A"
        match = (expected == pred_label)

        if is_detected and is_hand_expected:
            detected_hand_images += 1
            if match:
                correct_count += 1
        elif not is_detected and not is_hand_expected:
            # 'nothing' correctly not detected
            match = True

        match_symbol = "[YES]" if match else "[NO]"

        print(f"{img_path.name:18} | {expected:8} | {str(is_detected):8} | {pred_label:9} | {conf_str:10} | {out.status_message[:20]:20} | {match_symbol}")

        results.append({
            "filename": img_path.name,
            "expected": expected,
            "hand_detected": is_detected,
            "predicted": pred_label,
            "confidence_score": out.confidence_score,
            "raw_probability": out.raw_probability,
            "margin": out.margin,
            "status_message": out.status_message,
            "match": match
        })

    print("=" * 80)
    print("                      BENCHMARK SUMMARY RESULTS                     ")
    print("=" * 80)
    det_rate = (detected_hand_images / max(1, total_hand_images)) * 100.0
    acc_rate = (correct_count / max(1, total_hand_images)) * 100.0
    det_acc = (correct_count / max(1, detected_hand_images)) * 100.0 if detected_hand_images > 0 else 0.0

    print(f"[+] Total Test Images Evaluated     : {len(image_files)}")
    print(f"[+] Hand Test Images Expected       : {total_hand_images} (excluding 'nothing' negative control)")
    print(f"[+] Hand Images Successfully Found  : {detected_hand_images} / {total_hand_images} ({det_rate:.2f}% Detection Rate)")
    print(f"[+] Correct Signs Recognized        : {correct_count} / {total_hand_images} ({acc_rate:.2f}% Overall Accuracy)")
    print(f"[+] Accuracy on Detected Hands      : {correct_count} / {detected_hand_images} ({det_acc:.2f}%)")
    print(f"[+] Negative Control ('nothing')    : {'PASS (No hand found)' if not results[-1]['hand_detected'] and results[-1]['expected'] == 'nothing' else 'TESTED'}")
    print("=" * 80)

    # Export report to CSV
    csv_out = Path("models/archive_test_benchmark.csv")
    pd.DataFrame(results).to_csv(csv_out, index=False)
    print(f"[+] Detailed report exported to: {csv_out}")


if __name__ == "__main__":
    evaluate_test_archive()
