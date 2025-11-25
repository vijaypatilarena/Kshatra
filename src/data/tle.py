from __future__ import annotations
from ast import parse
import time
import requests
from typing import List, Tuple

def fetch_tle_text(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def parse_tles(tle_text: str):
    """
    Convert TLE text into a list of (name, line1, line2) tuples.
    Skips malformed or incomplete TLEs safely.
    """
    lines = [ln.strip() for ln in tle_text.splitlines() if ln.strip()]
    tles = []
    i = 0
    while i < len(lines):
        # Check for 3-line format (Name + L1 + L2)
        if not lines[i].startswith("1 ") and i + 2 < len(lines):
            name = lines[i]
            l1, l2 = lines[i + 1], lines[i + 2]
            i += 3
        # Check for 2-line format
        elif i + 1 < len(lines) and lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            name = "UNKNOWN"
            l1, l2 = lines[i], lines[i + 1]
            i += 2
        else:
            # Skip incomplete or malformed TLE
            i += 1
            continue
        tles.append((name, l1, l2))
    return tles

def get_active_tles() -> List[Tuple[str, str, str]]:
    #This function and link helps to track and fetch data from current active satellites.
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    return parse_tles(fetch_tle_text(url))

def get_debris_tles() -> List[Tuple[str, str, str]]:
    # This function and link helps to track and fetch tle for a known debris group.
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=1980-009&FORMAT=tle"
    return parse_tles(fetch_tle_text(url))

if __name__ == "__main__":
    print("Fetching active satellites from ClesTrak ....")
    tles =get_active_tles()
    print(f"Fetched {len(tles)} TLE Records")
    print("Example TLE:")
    for t in tles[1]:
        print("\n".join(t))
