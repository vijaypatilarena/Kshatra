# scripts/merge_real_and_synthetic.py
"""
Merge real snapshot(s) and synthetic dataset together to produce training_merged.parquet.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from glob import glob

DATA_PATH = "data"
OUT_FILE = os.path.join(DATA_PATH, "training_merged.parquet")

def find_snapshot_files():
    return sorted(glob(os.path.join(DATA_PATH, "training_snapshot_*.parquet")), reverse=True)

def run_merge(synthetic_file=None):
    snapshots = find_snapshot_files()
    if not snapshots:
        print("No snapshot files found. Please run generate_training.py first.")
        return
    # load latest snapshot
    df_snap = pd.read_parquet(snapshots[0])
    print(f"Loaded snapshot {snapshots[0]} with {len(df_snap)} rows")
    # load synthetic
    if synthetic_file is None:
        # prefer parquet if exists
        pf = os.path.join(DATA_PATH, "synthetic_nearmiss.parquet")
        cf = os.path.join(DATA_PATH, "synthetic_nearmiss.csv")
        if os.path.exists(pf):
            synthetic_file = pf
        elif os.path.exists(cf):
            synthetic_file = cf
        else:
            print("No synthetic file found; aborting merge.")
            return
    if synthetic_file.endswith(".parquet"):
        df_syn = pd.read_parquet(synthetic_file)
    else:
        df_syn = pd.read_csv(synthetic_file)
    print(f"Loaded synthetic {synthetic_file} with {len(df_syn)} rows")
    # unify columns (some fields may differ)
    df_all = pd.concat([df_snap, df_syn], ignore_index=True, sort=False)
    df_all = df_all.reset_index(drop=True)
    df_all.to_parquet(OUT_FILE, index=False)
    print(f"Wrote merged dataset to {OUT_FILE} ({len(df_all)} rows)")
    return df_all

if __name__ == "__main__":
    run_merge()
