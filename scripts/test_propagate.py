
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from src.data.tle import get_active_tles

tles = get_active_tles()
name, l1, l2 = tles[0]
sat = Satrec.twoline2rv(l1, l2)

start = datetime.utcnow()
print(f"Propagating: {name}")
for minutes in range(0, 300, 60):
    dt = start + timedelta(minutes=minutes)
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    err, r, v = sat.sgp4(jd, fr)
    print(f"{minutes:>3} min → Error={err} Pos(km)={r} Vel(km/s)={v}")
