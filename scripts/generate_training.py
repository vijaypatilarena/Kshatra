# scripts/generate_training.py
"""
Small training data generator:
 - takes a snapshot (subset) of TLEs
 - coarsely screens with KD-tree (configurable)
 - refines each candidate to compute TCA, miss distance, relative speed
 - produces a DataFrame and saves to data/training_snapshot_*.parquet

Additionally:
 - can merge synthetic CSV (data/synthetic_nearmiss.csv) into the snapshot
 - create a balanced dataset via SMOTE (if available) or simple oversampling
"""
import datetime as dt
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sgp4.api import Satrec
from src.data.tle import get_active_tles, get_debris_tles
from src.propagation.sgp4_propagator import time_grid
from src.screening.coarse import coarse_screen
from src.screening.refine import closest_approach, _pos_vel_from_sat
from src.features.encounter_feature import pair_features

OUT_PATH = "data"
os.makedirs(OUT_PATH, exist_ok=True)


def run_snapshot(now: dt.datetime, max_active=200, max_debris=200,
                 coarse_step_s=300, coarse_radius_km=50, fine_window_s=600, fine_step_s=5):
    active = get_active_tles()[:max_active]
    debris = get_debris_tles()[:max_debris]
    all_tles = active + debris
    print(f"Using {len(all_tles)} objects (active={len(active)} debris={len(debris)})")

    # Build Satrec objects
    sats = {}
    for i, (_, l1, l2) in enumerate(all_tles):
        try:
            sat = Satrec.twoline2rv(l1, l2)
            sats[i] = sat
        except Exception as e:
            print(f"Skipping sat {i} due to parse error: {e}")

    # Coarse grid (for screening)
    jd_grid = time_grid(now, hours=48, step_s=coarse_step_s)

    # Propagate positions for coarse grid
    positions = {}
    for i, sat in sats.items():
        pos = []
        for jd in jd_grid:
            jd_i = int(np.floor(jd))
            jd_f = float(jd - jd_i)
            err, r, v = sat.sgp4(jd_i, jd_f)
            if err != 0:
                pos.append([np.nan, np.nan, np.nan])
            else:
                pos.append(r)
        positions[i] = np.array(pos)

    # Coarse screening
    candidates = coarse_screen(positions, dist_km=coarse_radius_km)
    print(f"Coarse candidates found: {len(candidates)}")

    rows = []
    for t_idx, a, b in candidates:
        try:
            jd_center = jd_grid[t_idx]
            tca_jd, miss_km, vrel = closest_approach(sats[a], sats[b], jd_center,
                                                     window_s=fine_window_s, step_s=fine_step_s)
            # compute r,v at TCA
            rA, vA = _pos_vel_from_sat(sats[a], tca_jd)
            rB, vB = _pos_vel_from_sat(sats[b], tca_jd)
            feats = pair_features(rA, vA, rB, vB)
            feats.update({
                "obj_a": int(a),
                "obj_b": int(b),
                "tca_jd": float(tca_jd),
                "miss_km": float(miss_km),
                "vrel_kms": float(vrel),
                "label": int(miss_km < 1.0),
            })
            rows.append(feats)
        except Exception as e:
            print(f"Refine failed for pair ({a},{b}) at t_idx {t_idx}: {e}")
            continue

    if rows:
        df = pd.DataFrame(rows)
        out_file = os.path.join(OUT_PATH, f"training_snapshot_{now.strftime('%Y%m%dT%H%M%SZ')}.parquet")
        df.to_parquet(out_file, index=False)
        print(f"Wrote {len(df)} candidate rows to {out_file}")
        return df
    else:
        print("No refined candidates to write")
        return pd.DataFrame()


# ----------------------------
# Merge & balancing helpers
# ----------------------------
def merge_with_synthetic(snapshot_df: pd.DataFrame, synthetic_path: str = "data/synthetic_nearmiss.parquet") -> pd.DataFrame:
    """Merge snapshot dataframe and synthetic dataset if available."""
    if os.path.exists(synthetic_path):
        synth = pd.read_parquet(synthetic_path)
        # harmonize column names: make sure both have same feature columns
        # drop meta keys that may differ, keep features+label
        # assume pair_features returns consistent column names
        print(f"Found synthetic dataset at {synthetic_path} ({len(synth)} rows). Merging.")
        merged = pd.concat([snapshot_df, synth], ignore_index=True, sort=False)
    else:
        print("No synthetic dataset found; skipping merge.")
        merged = snapshot_df
    # drop duplicates conservatively
    merged = merged.drop_duplicates().reset_index(drop=True)
    out = os.path.join(OUT_PATH, "training_merged.parquet")
    merged.to_parquet(out, index=False)
    print(f"Wrote merged dataset to {out} ({len(merged)} rows)")
    return merged


def balance_dataset(df: pd.DataFrame, method: str = "smote") -> pd.DataFrame:
    """
    Try SMOTE balancing first (imblearn). If not available or fails,
    fallback to naive oversampling of positive class.
    Returns balanced dataframe and writes to disk.
    """
    from collections import Counter
    print("Class distribution before balancing:", Counter(df["label"]))
    X = df.drop(columns=["label"])
    y = df["label"]

    balanced_df = None

    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            balanced_df = pd.DataFrame(X_res, columns=X.columns)
            balanced_df["label"] = y_res
            print("SMOTE completed.")
        except Exception as e:
            print("SMOTE failed or imblearn not installed:", e)
            method = "oversample"

    if method == "oversample":
        # naive oversample minority class to majority
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        if len(pos) == 0:
            print("No positive samples to oversample — returning original dataset.")
            balanced_df = df.copy()
        else:
            # oversample pos to match neg
            n_rep = int(np.ceil(len(neg) / len(pos)))
            pos_up = pd.concat([pos] * n_rep, ignore_index=True)
            pos_up = pos_up.sample(n=len(neg), replace=True, random_state=42)
            balanced_df = pd.concat([neg, pos_up], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
            print("Naive oversampling completed.")

    # write balanced
    out_bal = os.path.join(OUT_PATH, "training_balanced.parquet")
    balanced_df.to_parquet(out_bal, index=False)
    print(f"Wrote balanced dataset to {out_bal} ({len(balanced_df)} rows)")
    from collections import Counter
    print("Class distribution after balancing:", Counter(balanced_df["label"]))

    # also plot class distribution
    try:
        cnt = Counter(balanced_df["label"])
        plt.figure(figsize=(4, 3))
        plt.bar([str(k) for k in cnt.keys()], [v for v in cnt.values()])
        plt.title("Class distribution (balanced)")
        plt.xlabel("label")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_PATH, "class_distribution_balanced.png"))
        plt.close()
    except Exception:
        pass

    return balanced_df


if __name__ == "__main__":
    now = dt.datetime.utcnow().replace(microsecond=0)
    snap_df = run_snapshot(now, max_active=200, max_debris=0, coarse_step_s=600, coarse_radius_km=50,
                           fine_window_s=600, fine_step_s=5)
    if not snap_df.empty:
        merged = merge_with_synthetic(snap_df, synthetic_path=os.path.join(OUT_PATH, "synthetic_nearmiss.parquet"))
        balanced = balance_dataset(merged, method="smote")
        print(balanced.head())
    else:
        print("No snapshot rows — nothing to merge or balance.")
