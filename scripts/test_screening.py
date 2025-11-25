from datetime import datetime
from src.data.tle import get_active_tles, get_debris_tles
from src.propagation.sgp4_propagator import build_satrec, time_grid, propagate_positions
from src.screening.coarse import coarse_screen
import numpy as np
active = get_active_tles()[:20]
debris = get_debris_tles()[:20]
all_tles = active + debris
sats = {i: build_satrec(l1, l2) for i, (_, l1, l2) in enumerate(all_tles)}

# Build time grid for 48 hours (5 min steps)
times_jd = time_grid(datetime.utcnow(), hours=48, step_s=600)

positions = {}
for i, sat in sats.items():
    positions[i] = propagate_positions(sat, times_jd)


close_pairs = coarse_screen(positions, dist_km=10)
print(f"Detected {len(close_pairs)} potential close approaches within 10 km")
if close_pairs:
    print("Sample:", close_pairs[:5])
