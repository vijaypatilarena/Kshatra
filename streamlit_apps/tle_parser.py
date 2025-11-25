import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sgp4.api import Satrec

def parse_tle_input(t1, t2):
    if not t1 or not t2:
        raise ValueError("Both TLE blocks required")

        l1 = t1.splitlines()[0].strip()
        l2 = t1.splitlines()[1].strip()

        m1 = t2.splitlines()[0].strip()
        m2 = t2.splitlines()[1].strip()

        satA = Satrec.twoline2rv(l1, l2)
        satB = Satrec.twolines2rv(m1, m2)

        return satA, satB