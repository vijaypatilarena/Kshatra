# scripts/predict_example.py
"""
Example script to load a trained model and run a prediction.
It loads feature names from training_balanced.parquet,
creates a random feature vector (or you can input your own),
and outputs predicted probability of collision.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_PATH = "models/collision_predictor_xgboost.pkl"

DATA_PATH = "data/training_balanced.parquet"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train using train_model.py first.")

    print(f"Loading model: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_features():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run balanced_training.py first.")

    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("label", "obj_a", "obj_b")]
    return feature_cols


def make_random_sample(feature_cols):
    """
    Creates a random but valid input sample using reasonable ranges
    derived from typical orbital mechanics.
    """
    sample = {}

    for col in feature_cols:
        if "miss" in col:
            sample[col] = float(np.random.uniform(-5, 5))
        elif "norm" in col:
            sample[col] = float(np.random.uniform(6500, 7500))  # Earth orbital radius-ish
        elif "vrel" in col:
            sample[col] = float(np.random.uniform(0, 15))
        elif "closure" in col:
            sample[col] = float(np.random.uniform(-10, 10))
        else:
            sample[col] = float(np.random.normal(0, 1))

    return sample


def main():
    model = load_model()
    feature_cols = load_features()

    print(f"Loaded {len(feature_cols)} feature columns.")
    print("Generating a random synthetic sample for demonstration...\n")

    sample = make_random_sample(feature_cols)
    X = pd.DataFrame([sample])[feature_cols]

    # Prediction
    pred_class = model.predict(X)[0]
    pred_prob = model.predict_proba(X)[0][1]  # probability of class=1 (collision risk)

    print("\n===== Prediction Result =====")
    print(f"Predicted class: {pred_class} (1 = collision risk)")
    print(f"Predicted probability: {pred_prob:.6f}")
    print("=============================\n")

    print("Sample used:")
    for k, v in sample.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
