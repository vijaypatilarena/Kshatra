# scripts/generate_synthetic.py
"""
Synthetic encounter generator (no SGP4)
---------------------------------------
- Guaranteed to produce synthetic near-miss + negative samples
- Uses your pair_features function from src/features/encounter_feature.py
- Writes data/synthetic_nearmiss.csv
- Print progress and final counts so you can see what happened
"""

import os
import sys
from typing import Tuple
import numpy as np
import pandas as pd

# ensure project root is on path when running as module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your feature builder (you said file is src/features/encounter_feature.py)
from src.features.encounter_feature import pair_features

OUT_PATH = "data"
OUT_FILE = os.path.join(OUT_PATH, "synthetic_nearmiss.csv")
os.makedirs(OUT_PATH, exist_ok=True)


def random_orbit_vector() -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a random orbit-like position (km) and velocity (km/s).
    Position radius chosen for LEO-like (Earth centre distance ~ 6500-7500 km).
    Velocity magnitude chosen ~7-8 km/s.
    """
    r_mag = np.random.uniform(6500, 7500)
    r = np.random.normal(size=3)
    r = r / np.linalg.norm(r) * r_mag

    v_mag = np.random.uniform(7.0, 8.0)
    v = np.random.normal(size=3)
    v = v / np.linalg.norm(v) * v_mag

    return r, v


def generate_synthetic_samples(n_samples: int = 2000,
                               positive_fraction: float = 0.4,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Generate synthetic encounter dataset.
    - n_samples: total rows
    - positive_fraction: fraction of rows labelled positive (near-miss)
    Returns DataFrame and writes CSV to data/synthetic_nearmiss.csv
    """

    if verbose:
        print(f"Generating {n_samples} synthetic orbital encounters (pos_frac={positive_fraction})...")

    rows = []
    num_pos = int(n_samples * positive_fraction)
    num_neg = n_samples - num_pos

    # positives: miss distances between 0.2 - 0.9 km
    for i in range(num_pos):
        rA, vA = random_orbit_vector()

        # create a small miss offset
        dir_vec = np.random.normal(size=3)
        dir_vec /= np.linalg.norm(dir_vec)
        miss = float(np.random.uniform(0.2, 0.9))  # km
        rB = rA + dir_vec * miss

        # small relative velocity perturbation
        vB = vA + np.random.normal(scale=0.01, size=3)

        try:
            feat = pair_features(rA, vA, rB, vB)
        except Exception as e:
            if verbose:
                print(f"Skipping positive sample {i}: pair_features() error: {e}")
            continue

        feat["miss_km"] = miss
        feat["vrel_kms"] = float(np.linalg.norm(vA - vB))
        feat["label"] = 1
        rows.append(feat)

    # negatives: miss distances between 5 - 50 km
    for i in range(num_neg):
        rA, vA = random_orbit_vector()
        dir_vec = np.random.normal(size=3)
        dir_vec /= np.linalg.norm(dir_vec)
        miss = float(np.random.uniform(5.0, 50.0))
        rB = rA + dir_vec * miss
        vB = vA + np.random.normal(scale=0.05, size=3)

        try:
            feat = pair_features(rA, vA, rB, vB)
        except Exception as e:
            if verbose:
                print(f"Skipping negative sample {i}: pair_features() error: {e}")
            continue

        feat["miss_km"] = miss
        feat["vrel_kms"] = float(np.linalg.norm(vA - vB))
        feat["label"] = 0
        rows.append(feat)

    df = pd.DataFrame(rows)

    # Shuffle rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df.to_csv(OUT_FILE, index=False)

    if verbose:
        print("---------------------------------------------------")
        print(f"Generated: {len(df)} synthetic samples")
        print(f"Positives: {int(df['label'].sum())}  Negatives: {len(df) - int(df['label'].sum())}")
        print(f"Saved → {OUT_FILE}")
        print("---------------------------------------------------")

    return df


if __name__ == "__main__":
    generate_synthetic_samples(n_samples=2000, positive_fraction=0.4, verbose=True)
