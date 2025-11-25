from __future__ import annotations
import numpy as np
from typing import Dict

def rt_n_frame(r: np.ndarray, v: np.ndarray) -> tuple:
    """
    Compute RTN (Radial, Transverse, Normal) unit vectors for position r and velocity v.
    r, v: numpy arrays (3,)
    Returns (rhat, that, nhat) each (3,) unit vectors.
    """
    rnorm = np.linalg.norm(r)
    if rnorm == 0:
        raise ValueError("Zero position vector")
    rhat = r / rnorm
    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)
    if hnorm == 0:
        nhat = np.array([0.0, 0.0, 1.0])
    else:
        nhat = h / hnorm
    that = np.cross(nhat, rhat)
    return rhat, that, nhat

def pair_features(rA: np.ndarray, vA: np.ndarray, rB: np.ndarray, vB: np.ndarray) -> Dict[str, float]:
    """
    Generate features for a close-approach event at TCA.
    Returns a dict with geometric and kinematic features.
    """
    dr = rB - rA
    dv = vB - vA
    rhat, that, nhat = rt_n_frame(rA, vA)

    miss_r = float(np.dot(dr, rhat))
    miss_t = float(np.dot(dr, that))
    miss_n = float(np.dot(dr, nhat))
    miss_norm = float(np.linalg.norm(dr))
    vrel = float(np.linalg.norm(dv))
    closure_rate = float(np.dot(dv, dr) / (miss_norm + 1e-12))  # +ve separating, -ve closing

    return {
        "miss_r": miss_r,
        "miss_t": miss_t,
        "miss_n": miss_n,
        "miss_norm": miss_norm,
        "vrel": vrel,
        "closure_rate": closure_rate,
        "rA_norm_km": float(np.linalg.norm(rA)),
        "rB_norm_km": float(np.linalg.norm(rB)),
    }
