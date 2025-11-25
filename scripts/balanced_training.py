# scripts/balanced_training.py
"""
Balance dataset using:
 - Simple oversampling of positive class
 - SMOTE (if imblearn is installed)
Outputs:
    data/training_balanced.parquet
    data/class_distribution_balanced.png
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_PATH = "data"
IN_FILE = os.path.join(DATA_PATH, "training_merged.parquet")
OUT_FILE = os.path.join(DATA_PATH, "training_balanced.parquet")
PLOT_FILE = os.path.join(DATA_PATH, "class_distribution_balanced.png")


# -----------------------
# Load merged dataset
# -----------------------
def load_merged():
    if not os.path.exists(IN_FILE):
        raise FileNotFoundError(
            f"{IN_FILE} not found. You must run merge_real_and_synthetic.py first."
        )
    return pd.read_parquet(IN_FILE)


# -----------------------
# Simple oversampling
# -----------------------
def oversample_simple(df, factor=5):
    df_pos = df[df["label"] == 1]
    df_neg = df[df["label"] == 0]

    if df_pos.empty:
        print("⚠ No positive samples found. Returning dataset unchanged.")
        return df

    df_pos_oversampled = df_pos.sample(
        len(df_pos) * factor,
        replace=True,
        random_state=42
    )

    df_bal = pd.concat([df_neg, df_pos_oversampled], ignore_index=True)
    df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_bal


# -----------------------
# Try SMOTE
# -----------------------
def try_smote(X, y):
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as e:
        print("⚠ imbalanced-learn (SMOTE) not installed:", e)
        return None

    try:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
    except Exception as e:
        print("⚠ SMOTE failed:", e)
        return None


# -----------------------
# Balance dataset
# -----------------------
def run_balance(use_smote=True, oversample_factor=5):
    df = load_merged()

    print("📊 Original distribution:")
    print(df["label"].value_counts())

    # split features and labels
    X = df.drop(columns=["label"])
    y = df["label"]

    # Try SMOTE first
    if use_smote:
        smote_result = try_smote(X, y)
        if smote_result is not None:
            print("✅ Using SMOTE balancing")
            X_res, y_res = smote_result
            df_balanced = pd.DataFrame(X_res, columns=X.columns)
            df_balanced["label"] = y_res
        else:
            print("⚠ SMOTE unavailable → Falling back to simple oversampling")
            df_balanced = oversample_simple(df, factor=oversample_factor)
    else:
        df_balanced = oversample_simple(df, factor=oversample_factor)

    # Save parquet
    df_balanced.to_parquet(OUT_FILE, index=False)
    print(f"💾 Saved balanced dataset → {OUT_FILE}")

    # Plot class distribution
    plt.figure(figsize=(4, 3))
    df_balanced["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Balanced Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()

    print(f"📈 Saved class distribution plot → {PLOT_FILE}")
    print("Done!")

    return df_balanced


if __name__ == "__main__":
    run_balance(use_smote=True, oversample_factor=5)
