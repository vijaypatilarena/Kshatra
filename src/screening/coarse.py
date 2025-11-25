# src/screening/coarse.py
from __future__ import annotations
import numpy as np
from sklearn.neighbors import KDTree
from typing import Dict, List, Tuple

def coarse_screen(positions_by_obj: Dict[int, np.ndarray],
                  dist_km: float = 10.0) -> List[Tuple[int, int, int]]:
    obj_ids = list(positions_by_obj.keys())
    T = next(iter(positions_by_obj.values())).shape[0]
    close_pairs = []
    for t in range(T):
        pts = np.vstack([positions_by_obj[i][t] for i in obj_ids])
        tree = KDTree(pts)
        pairs = tree.query_radius(pts, r=dist_km)
        for i, neigh in enumerate(pairs):
            for j in neigh:
                if j <= i:
                    continue
                close_pairs.append((t, obj_ids[i], obj_ids[j]))
    return close_pairs
