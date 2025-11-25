# src/propagation/sgp4_propagator.py
from __future__ import annotations
import numpy as np
from datetime import datetime
from sgp4.api import Satrec, jday

def build_satrec(line1: str, line2: str) -> Satrec:
    return Satrec.twoline2rv(line1, line2)

def time_grid(start_time: datetime, hours=48, step_s=300) -> np.ndarray:
    jd_start, fr_start = jday(
        start_time.year, start_time.month, start_time.day,
        start_time.hour, start_time.minute, start_time.second
    )
    step_days = step_s / 86400.0
    num_steps = int(hours * 3600 / step_s) + 1
    return jd_start + fr_start + np.arange(num_steps) * step_days

def propagate_positions(sat: Satrec, times_jd: np.ndarray) -> np.ndarray:
    pos = []
    for jd in times_jd:
        jd_i = int(np.floor(jd))
        jd_f = jd - jd_i
        e, r, _ = sat.sgp4(jd_i, jd_f)
        pos.append(r if e == 0 else [np.nan, np.nan, np.nan])
    return np.array(pos)
