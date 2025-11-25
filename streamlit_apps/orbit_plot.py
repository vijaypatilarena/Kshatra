import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import plotly.graph_objects as go
from sgp4.api import jday
from src.screening.refine import _pos_vel_from_sat


def plot_orbits_3d(satA, satB, tca_jd):
    """
    Plot 3D orbits of Satellite A and B around the TCA (±20 minutes window).
    Returns a Plotly Figure object.
    """

    # Time window: ±20 minutes around TCA
    times = np.linspace(tca_jd - 20/1440, tca_jd + 20/1440, 300)

    ptsA = []
    ptsB = []

    # Propagate both satellites
    for jd in times:
        rA, _ = _pos_vel_from_sat(satA, jd)
        rB, _ = _pos_vel_from_sat(satB, jd)
        ptsA.append(rA)
        ptsB.append(rB)

    ptsA = np.array(ptsA)
    ptsB = np.array(ptsB)

    # ----------- Plotly Figure -----------
    fig = go.Figure()

    # Orbit A
    fig.add_trace(go.Scatter3d(
        x=ptsA[:, 0], y=ptsA[:, 1], z=ptsA[:, 2],
        mode="lines",
        name="Satellite A",
        line=dict(color="#00c3ff", width=4)
    ))

    # Orbit B
    fig.add_trace(go.Scatter3d(
        x=ptsB[:, 0], y=ptsB[:, 1], z=ptsB[:, 2],
        mode="lines",
        name="Satellite B",
        line=dict(color="#ff4f4f", width=4)
    ))

    # Earth Sphere
    Re = 6371  # km
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x = Re * np.cos(u) * np.sin(v)
    y = Re * np.sin(u) * np.sin(v)
    z = Re * np.cos(v)

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale="Blues",
        showscale=False,
        opacity=0.6,
        name="Earth"
    ))

    # Layout
    fig.update_layout(
        title="3D Orbit Visualization",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data",
            bgcolor="#111"
        ),
        paper_bgcolor="#111",
        font=dict(color="white"),
        height=650,
    )

    return fig
