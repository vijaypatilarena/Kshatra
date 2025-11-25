# scripts/generate_synthetic.py
"""
Generate synthetic near-miss samples by perturbing TLE-derived Satrec objects.
Saves CSV + Parquet -> data/synthetic_nearmiss.csv / .parquet
This version:
 - Uses original TLE text when available (preferred)
 - Creates targeted positives by gradually increasing perturbation until a near-miss is found
 - Produces a balanced mix if requested (pos_frac argument)
"""
from __future__ import annotations
import os
import sys
import time
import random
from typing import List, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday
from src.data.tle import get_active_tles
from src.screening.refine import closest_approach, _pos_vel_from_sat
from src.features.encounter_feature import pair_features  # your file: encounter_feature.py

OUT_PATH = "data"
os.makedirs(OUT_PATH, exist_ok=True)


def random_perturb(scale: float = 1e-3) -> float:
    """Small uniform perturbation (radians or mean anomaly in rad)."""
    return float(np.random.uniform(-scale, scale))


def clone_and_perturb_from_tle(l1: str, l2: str,
                              incl_delta=0.0, raan_delta=0.0,
                              argp_delta=0.0, mo_delta=0.0,
                              ndot_delta=0.0, nddot_delta=0.0) -> Optional[Satrec]:
    """
    Create a new Satrec by parsing the original TLE and applying small perturbations
    to orbital elements (done by editing the TLE strings and re-parsing).
    If editing the TLE lines is unreliable, we return None.
    """
    try:
        # Parse original
        sat = Satrec.twoline2rv(l1, l2)
    except Exception:
        return None

    # Many TLE fields are encoded in fixed-width numeric formats.
    # We'll attempt a pragmatic approach: modify the mean anomaly / arg of perigee / inclination/RAAN
    # by re-writing the numeric fields in the TLE lines. Keep safe: if formatting fails, return None.

    try:
        # helper to replace a field in a TLE line by slice indices (approx, tolerant)
        def write_field(line: str, start: int, end: int, fmt_val: str) -> str:
            return line[:start] + fmt_val + line[end:]

        # Make safe copies
        l1s = l1.rstrip("\n")
        l2s = l2.rstrip("\n")

        # Extract numeric fields by positions (approx typical TLE layout) — tolerant: may fail for some producers.
        # mean anomaly (degrees) usually at columns 43-51 in line 2; arg of perigee 34-42; inclination 8-16; raan 17-25
        incl = float(l2s[8:16].strip())
        raan = float(l2s[17:25].strip())
        argp = float(l2s[34:42].strip())
        mo = float(l2s[43:51].strip())

        # apply deltas (deltas are in degrees for TLE numeric fields)
        incl_new = (incl + np.degrees(incl_delta)) % 360.0
        raan_new = (raan + np.degrees(raan_delta)) % 360.0
        argp_new = (argp + np.degrees(argp_delta)) % 360.0
        mo_new = (mo + np.degrees(mo_delta)) % 360.0

        # reformat fields with the same widths
        l2s_new = l2s
        l2s_new = write_field(l2s_new, 8, 16, f"{incl_new:8.4f}")
        l2s_new = write_field(l2s_new, 17, 25, f"{raan_new:8.4f}")
        l2s_new = write_field(l2s_new, 34, 42, f"{argp_new:8.4f}")
        l2s_new = write_field(l2s_new, 43, 51, f"{mo_new:8.4f}")

        # Parse back to Satrec
        sat2 = Satrec.twoline2rv(l1s, l2s_new)
        return sat2
    except Exception:
        return None


def tweak_satrec_direct(sat: Satrec,
                        incl_delta=0.0, raan_delta=0.0,
                        argp_delta=0.0, mo_delta=0.0) -> Optional[Satrec]:
    """
    If we don't have TLE strings, attempt to clone Satrec and set numeric fields directly.
    Works for Satrec objects that expose common fields (inclo, raan, argpo, mo).
    """
    try:
        sat2 = Satrec()
        # copy safe known fields if present
        for attr in ("satnum", "epochyr", "epochdays", "ndot", "nddot", "bstar",
                     "inclo", "raan", "ecco", "argpo", "mo", "no_kozai"):
            if hasattr(sat, attr):
                setattr(sat2, attr, getattr(sat, attr))
        # apply perturbations
        if hasattr(sat2, "inclo"):
            sat2.inclo += incl_delta
        if hasattr(sat2, "raan"):
            sat2.raan += raan_delta
        if hasattr(sat2, "argpo"):
            sat2.argpo += argp_delta
        if hasattr(sat2, "mo"):
            sat2.mo += mo_delta
        return sat2
    except Exception:
        return None


def generate_synthetic_samples(n_samples: int = 1000,
                               pos_frac: float = 0.5,
                               max_attempts: int = 5000,
                               jd_center: Optional[float] = None):
    """
    Generate synthetic samples; return DataFrame.
    pos_frac: fraction of samples that should be 'positive' (label=1)
    max_attempts: maximum tries to find positives (will stop earlier if not possible)
    """
    print(f"Generating {n_samples} synthetic samples...")
    tles = get_active_tles()  # expects list of tuples (satnum, l1, l2) OR list of Satrec objects
    sats_src = []
    # normalize to list of (Satrec, l1, l2) if possible
    for entry in tles:
        if isinstance(entry, tuple) and len(entry) >= 3:
            try:
                sat = Satrec.twoline2rv(entry[1], entry[2])
                sats_src.append((sat, entry[1], entry[2]))
            except Exception:
                # fallback: maybe entry already Satrec-like? skip if cannot parse
                continue
        elif isinstance(entry, Satrec):
            sats_src.append((entry, None, None))
        else:
            # unknown format — skip
            continue

    if not sats_src:
        print("No usable Satrec inputs found from get_active_tles(); exiting.")
        return

    print(f"Using {len(sats_src)} input Satrec objects to synthesize from")
    # use approximate JD center now if not passed; compute from first sat epoch (if available)
    if jd_center is None:
        # attempt to compute a jd centre using current UTC
        from datetime import datetime
        d = datetime.utcnow()
        jd_center = float(jday(d.year, d.month, d.day, d.hour, d.minute, d.second)[0] +
                          jday(d.year, d.month, d.day, d.hour, d.minute, d.second)[1])
    print(f"jd_center (approx): {jd_center:.6f}")

    rows = []
    target_pos = int(n_samples * pos_frac)
    target_neg = n_samples - target_pos

    pos_count = 0
    neg_count = 0
    attempts = 0

    # Strategy:
    # - For positives: pick a real sat A, create B by slight perturbation (gradually increase perturbation to find near-miss)
    # - For negatives: pick two random distinct sats (or perturb heavily)
    while (pos_count < target_pos or neg_count < target_neg) and attempts < max_attempts:
        attempts += 1
        # pick source sat
        src_idx = random.randrange(len(sats_src))
        A, l1, l2 = sats_src[src_idx]

        # Decide positive vs negative to target distribution
        target_positive = pos_count < target_pos
        if target_positive:
            # try to create B by gradually increasing perturbation until close approach < threshold
            found = False
            # start with very small perturb and if not found, increase
            for scale in (1e-5, 5e-5, 2e-4, 8e-4, 2e-3, 5e-3):
                # attempt to create B
                if l1 and l2:
                    B = clone_and_perturb_from_tle(l1, l2,
                                                   incl_delta=random_perturb(scale),
                                                   raan_delta=random_perturb(scale),
                                                   argp_delta=random_perturb(scale),
                                                   mo_delta=random_perturb(scale * 4))
                else:
                    B = tweak_satrec_direct(A,
                                            incl_delta=random_perturb(scale),
                                            raan_delta=random_perturb(scale),
                                            argp_delta=random_perturb(scale),
                                            mo_delta=random_perturb(scale * 4))

                if B is None:
                    continue

                try:
                    tca_jd, miss_km, vrel = closest_approach(A, B, jd_center,
                                                            window_s=600, step_s=5)
                except Exception:
                    continue

                # Positive if miss under 1.0 km (tunable)
                if miss_km < 1.0:
                    # compute r,v and features
                    rA, vA = _pos_vel_from_sat(A, tca_jd)
                    rB, vB = _pos_vel_from_sat(B, tca_jd)
                    feats = pair_features(rA, vA, rB, vB)
                    feats.update({
                        "miss_km": float(miss_km),
                        "vrel_kms": float(vrel),
                        "label": 1,
                    })
                    rows.append(feats)
                    pos_count += 1
                    found = True
                    break
            if not found:
                # if couldn't find a true near-miss within tries, optionally accept a close-but-not-close sample as negative
                # fallback to negative branch
                choice_negative = True
            else:
                choice_negative = False
        else:
            choice_negative = True

        if choice_negative:
            # create a benign negative sample
            # pick another random sat distinct from src_idx
            other_idx = src_idx
            while other_idx == src_idx:
                other_idx = random.randrange(len(sats_src))
            C, l1c, l2c = sats_src[other_idx]

            # lightly perturb C to create variation
            B = None
            if l1c and l2c:
                B = clone_and_perturb_from_tle(l1c, l2c,
                                               incl_delta=random_perturb(1e-3),
                                               raan_delta=random_perturb(1e-3),
                                               argp_delta=random_perturb(1e-3),
                                               mo_delta=random_perturb(5e-3))
            if B is None:
                B = tweak_satrec_direct(C,
                                        incl_delta=random_perturb(1e-3),
                                        raan_delta=random_perturb(1e-3),
                                        argp_delta=random_perturb(1e-3),
                                        mo_delta=random_perturb(5e-3))
            # evaluate
            try:
                tca_jd, miss_km, vrel = closest_approach(A, B, jd_center,
                                                        window_s=600, step_s=5)
            except Exception:
                continue
            # mark negative if miss > 5 km (tunable) else accept small fraction as negatives anyway
            if miss_km > 5.0 or neg_count < target_neg:
                # compute feats
                try:
                    rA, vA = _pos_vel_from_sat(A, tca_jd)
                    rB, vB = _pos_vel_from_sat(B, tca_jd)
                except Exception:
                    continue
                feats = pair_features(rA, vA, rB, vB)
                feats.update({
                    "miss_km": float(miss_km),
                    "vrel_kms": float(vrel),
                    "label": 0,
                })
                rows.append(feats)
                neg_count += 1

        # progress print occasionally
        if (pos_count + neg_count) % 200 == 0:
            print(f"Generated {pos_count + neg_count}/{n_samples} samples (pos={pos_count}, neg={neg_count})")

    total = len(rows)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_PATH, "synthetic_nearmiss.csv")
    out_parquet = os.path.join(OUT_PATH, "synthetic_nearmiss.parquet")
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)
    print(f"Attempts: {attempts} → Generated {total} samples (positive={pos_count} negative={neg_count})")
    print(f"✅ Generated {total} synthetic samples → {out_csv} (and parquet at {out_parquet})")
    return df


if __name__ == "__main__":
    # default: 1000 balanced 50/50
    generate_synthetic_samples(n_samples=1000, pos_frac=0.5, max_attempts=5000)
