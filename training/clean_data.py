"""
Phase 1: Dataset Preparation - Data Cleaning and Validation.

Uses Pandas to clean the extracted landmark dataset:
- Removes missing, NaN, and corrupt records.
- Identifies and removes duplicate records.
- Validates 63 coordinate feature bounds.
- Analyzes and reports class distribution.
- Exports the verified dataset for machine learning model training.
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.landmarks_util import get_feature_names, TOTAL_FEATURES


def clean_landmark_dataset(
    input_csv: str = "data/landmarks.csv",
    output_csv: str = "data/landmarks_cleaned.csv"
) -> pd.DataFrame:
    """
    Cleans, verifies, and formats the landmark dataset using Pandas.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input landmark CSV '{input_csv}' not found. "
            "Run 'python training/extract_landmarks.py' or 'python training/generate_sample_data.py' first."
        )

    print(f"\n=======================================================")
    print(f"       CLEANING ASL LANDMARK DATASET: {input_csv}       ")
    print(f"=======================================================")

    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"[*] Initial total records loaded: {initial_count}")

    # Step 1: Verify presence of label and feature columns
    feature_cols = get_feature_names()
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required feature columns: {missing_cols[:5]}...")

    if "label" not in df.columns:
        raise ValueError("Dataset is missing required 'label' column.")

    # Step 2: Drop missing / null values
    null_rows = df.isnull().any(axis=1).sum()
    if null_rows > 0:
        print(f"[-] Dropping {null_rows} rows containing missing/null values...")
        df = df.dropna().reset_index(drop=True)
    else:
        print("[+] No null/missing values found.")

    # Step 3: Remove duplicate feature rows (preserving negative control 'nothing' baseline samples)
    is_nothing = df["label"] == "nothing"
    non_nothing = df[~is_nothing]
    nothing_df = df[is_nothing]

    dup_mask = non_nothing.duplicated(subset=feature_cols)
    duplicate_rows = dup_mask.sum()
    if duplicate_rows > 0:
        print(f"[-] Dropping {duplicate_rows} duplicate landmark records...")
        non_nothing = non_nothing[~dup_mask]
    else:
        print("[+] No duplicate records found.")
    df = pd.concat([non_nothing, nothing_df], ignore_index=True)

    # Step 4: Ensure numeric feature datatypes and finite values
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any rows that became NaN due to invalid conversion
    invalid_numeric = df[feature_cols].isnull().any(axis=1).sum()
    if invalid_numeric > 0:
        print(f"[-] Dropping {invalid_numeric} rows with non-numeric values...")
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # Check finite numbers (inf / -inf)
    finite_mask = np.isfinite(df[feature_cols].values).all(axis=1)
    if not finite_mask.all():
        num_infinite = (~finite_mask).sum()
        print(f"[-] Dropping {num_infinite} rows with infinite values...")
        df = df[finite_mask].reset_index(drop=True)

    # Step 5: Check normalized coordinate range sanity [-2.5, 2.5]
    out_of_bounds = ((df[feature_cols] < -2.5) | (df[feature_cols] > 2.5)).any(axis=1).sum()
    if out_of_bounds > 0:
        print(f"[-] Dropping {out_of_bounds} rows with out-of-bounds coordinates...")
        df = df[~((df[feature_cols] < -2.5) | (df[feature_cols] > 2.5)).any(axis=1)].reset_index(drop=True)

    # Step 6: Clean label formatting
    df["label"] = df["label"].astype(str).str.strip()

    final_count = len(df)
    removed_count = initial_count - final_count

    print(f"\n[*] Cleaning Summary:")
    print(f"    - Initial rows : {initial_count}")
    print(f"    - Rows removed : {removed_count} ({removed_count / max(1, initial_count):.1%})")
    print(f"    - Final rows   : {final_count}")

    # Step 7: Class Distribution Analysis
    print(f"\n[*] Class Distribution ({df['label'].nunique()} unique classes):")
    class_counts = df["label"].value_counts().sort_index()
    for label, count in class_counts.items():
        pct = (count / final_count) * 100
        bar = "#" * int(pct // 2)
        print(f"    Class {str(label):<8}: {count:>5} samples ({pct:>5.1f}%) | {bar}")

    # Warn if any class has fewer than 10 samples
    min_count = class_counts.min()
    if min_count < 10:
        sparse_classes = class_counts[class_counts < 10].index.tolist()
        print(f"\n[!] WARNING: Classes with fewer than 10 samples: {sparse_classes}")
        print("    Consider adding more samples for robust classification.")

    # Save cleaned CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[+] Cleaned dataset successfully saved to: {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and validate ASL landmarks dataset.")
    parser.add_argument("--input", type=str, default="data/landmarks.csv", help="Raw input CSV path")
    parser.add_argument("--output", type=str, default="data/landmarks_cleaned.csv", help="Cleaned output CSV path")
    args = parser.parse_args()

    clean_landmark_dataset(input_csv=args.input, output_csv=args.output)
