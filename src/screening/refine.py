from __future__ import annotations
import numpy as np
from typing import Tuple
from sgp4.api import Satrec


def _pos_vel_from_sat(sat: Satrec, jd: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate a satellite using SGP4 to get position and velocity vectors at a given Julian Date.
    
    Args:
        sat (Satrec): Satellite record (from sgp4.api.Satrec.twoline2rv)
        jd (float): Julian date at which to propagate
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Position [km] and velocity [km/s]
    """
    jd_i = int(np.floor(jd))
    jd_f = float(jd - jd_i)
    err, r, v = sat.sgp4(jd_i, jd_f)
    if err != 0:
        raise RuntimeError(f"SGP4 error code {err}")
    return np.array(r, dtype=float), np.array(v, dtype=float)


def closest_approach(
    satA: Satrec,
    satB: Satrec,
    jd_center: float,
    window_s: int = 600,
    step_s: int = 5
) -> Tuple[float, float, float]:
    """
    Finds the Time of Closest Approach (TCA) between two satellites
    within ±window_s seconds of jd_center.
    
    Args:
        satA, satB (Satrec): Satellite objects to compare
        jd_center (float): Central Julian date for the search
        window_s (int): Half-window in seconds to search around jd_center
        step_s (int): Step size in seconds for fine sampling
    
    Returns:
        Tuple[float, float, float]: (TCA_JulianDate, MissDistance_km, RelativeSpeed_km/s)
    """
    # Build a fine time grid around the center point
    n_steps = int((2 * window_s) // step_s) + 1
    dt_days = (np.arange(n_steps) * step_s - window_s) / 86400.0
    times = jd_center + dt_days

    # Prepare arrays
    dists = np.empty(n_steps, dtype=float)
    vrels = np.empty(n_steps, dtype=float)

    # Propagate both satellites at each time
    for idx, jd in enumerate(times):
        try:
            rA, vA = _pos_vel_from_sat(satA, jd)
            rB, vB = _pos_vel_from_sat(satB, jd)
        except RuntimeError:
            dists[idx] = np.nan
            vrels[idx] = np.nan
            continue

        dr = rB - rA
        dv = vB - vA
        dists[idx] = np.linalg.norm(dr)
        vrels[idx] = np.linalg.norm(dv)

    # Remove any NaN entries
    valid = ~np.isnan(dists)
    if not np.any(valid):
        raise RuntimeError("No valid propagation points in window")

    # Find minimum distance and corresponding time
    min_idx = np.nanargmin(dists)
    tca_jd = times[min_idx]
    miss_km = float(dists[min_idx])
    rel_speed_kms = float(vrels[min_idx])

    # Optional fine interpolation for smoother TCA estimation
    if 0 < min_idx < n_steps - 1:
        y0, y1, y2 = dists[min_idx - 1], dists[min_idx], dists[min_idx + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            offset = 0.5 * (y0 - y2) / denom  # parabolic interpolation offset
            tca_jd = tca_jd + (offset * step_s) / 86400.0
            try:
                rA, vA = _pos_vel_from_sat(satA, tca_jd)
                rB, vB = _pos_vel_from_sat(satB, tca_jd)
                miss_km = float(np.linalg.norm(rB - rA))
                rel_speed_kms = float(np.linalg.norm(vB - vA))
            except RuntimeError:
                pass  # fallback to coarse values if fine propagation fails

    return tca_jd, miss_km, rel_speed_kms
