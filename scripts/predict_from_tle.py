# scripts/predict_from_tle.py
"""
Predict collision risk between two satellites using their TLEs.
Usage:
    poetry run python -m scripts.predict_from_tle
"""

import sys
import os
import argparse
import numpy as np
from sgp4.api import Satrec, jday

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from src.screening.refine import closest_approach, _pos_vel_from_sat
from src.features.encounter_feature import pair_features


MODEL_PATH = "models/collision_predictor_xgboost.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\nRun: poetry run python -m scripts.train_model --model xgboost"
        )
    print(f"📦 Loading model → {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def read_tle_file(path: str):
    """Reads a file containing exactly 2 TLE lines."""
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"File {path} does not contain 2 TLE lines.")
    return lines[0], lines[1]


def predict_collision(tle1_l1, tle1_l2, tle2_l1, tle2_l2, model):
    satA = Satrec.twoline2rv(tle1_l1, tle1_l2)
    satB = Satrec.twoline2rv(tle2_l1, tle2_l2)

    # Use the epoch from the first TLE as reference
    jd0 = satA.jdsatepoch + satA.jdsatepochF

    # Compute closest approach between satA and satB
    tca_jd, miss_km, vrel_kms = closest_approach(
        satA, satB, jd_center=jd0, window_s=3600, step_s=10
    )

    # Get r and v at TCA
    rA, vA = _pos_vel_from_sat(satA, tca_jd)
    rB, vB = _pos_vel_from_sat(satB, tca_jd)

    # Extract features
    feats = pair_features(rA, vA, rB, vB)
    feats.update({
        "tca_jd": float(tca_jd),
        "miss_km": float(miss_km),
        "vrel_kms": float(vrel_kms),
    })

    # Convert to model order
    X = np.array([feats[col] for col in model.get_booster().feature_names], dtype=float).reshape(1, -1)

    # Predict
    prob = float(model.predict_proba(X)[0][1])
    label = int(prob >= 0.5)

    return label, prob, feats, tca_jd, miss_km, vrel_kms


def main():
    parser = argparse.ArgumentParser(description="Predict collision risk between two TLEs.")
    parser.add_argument("--tle1", type=str, help="Path to first TLE file.")
    parser.add_argument("--tle2", type=str, help="Path to second TLE file.")
    args = parser.parse_args()

    model = load_model()

    if args.tle1 and args.tle2:
        tle1_l1, tle1_l2 = read_tle_file(args.tle1)
        tle2_l1, tle2_l2 = read_tle_file(args.tle2)
    else:
        print("No TLEs provided → Using example ISS vs debris test")
        tle1_l1 = "1 25544U 98067A   24025.51782528  .00016807  00000+0  30865-3 0  9993"
        tle1_l2 = "2 25544  51.6401 353.5632 0004517  54.6776  44.8501 15.49212953437241"

        tle2_l1 = "1 40927U 15049C   24025.79876543  .00000023  00000+0  85612-4 0  9990"
        tle2_l2 = "2 40927  98.7100 218.9120 0012450  82.3456 277.8892 14.19502434500021"

    label, prob, feats, tca_jd, miss_km, vrel_kms = predict_collision(
        tle1_l1, tle1_l2, tle2_l1, tle2_l2, model
    )

    print("\n========== Collision Prediction ==========")
    print(f"TCA (JD):       {tca_jd}")
    print(f"Miss distance:  {miss_km:.3f} km")
    print(f"Relative speed: {vrel_kms:.3f} km/s")
    print("------------------------------------------")
    print(f"Predicted class: {label} (1 = collision risk)")
    print(f"Probability:     {prob:.6f}")
    print("==========================================\n")

    print("Features used:")
    for k, v in feats.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
