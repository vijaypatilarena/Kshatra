# streamlit_apps/app.py
"""
Kṣatra – Collision Risk Prediction Dashboard

Features:
- Live TLE input (Satellite A & B)
- TLE parsing & validation
- SGP4 propagation + closest approach computation
- Feature generation (pair_features)
- XGBoost collision-risk model prediction
- 3D orbit visualization around TCA
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, timezone

# --- Make repo importable (so we can import src.* and orbit_plot) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from sgp4.api import Satrec, jday
from src.screening.refine import closest_approach, _pos_vel_from_sat
from src.features.encounter_feature import pair_features

# orbit_plot.py must live in the same folder as this file
from orbit_plot import plot_orbits_3d  # type: ignore

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

MODEL_PATH = os.path.join(BASE_DIR, "models", "collision_predictor_xgboost.pkl")
FEATURE_COLUMNS = [
    "miss_r",
    "miss_t",
    "miss_n",
    "miss_norm",
    "vrel",
    "closure_rate",
    "rA_norm_km",
    "rB_norm_km",
    "tca_jd",
    "miss_km",
    "vrel_kms",
]

# Reasonable default TLEs (ISS + some example object)
DEFAULT_TLE_A = """1 25544U 98067A   21275.51001157  .00001264  00000-0  29621-4 0  9993
2 25544  51.6449  43.8797 0003437  97.7493  40.0133 15.48815396299327"""

DEFAULT_TLE_B = """1 43013U 17073A   21275.46914815  .00000133  00000-0  00000-0 0  9990
2 43013  97.6664 358.8629 0001379  96.3943 263.7450 14.91602063202254"""


@dataclass
class PredictionResult:
    tca_jd: float
    miss_km: float
    vrel_kms: float
    prob: float
    label: int
    features: Dict[str, float]


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}.\n"
            "Train it first using: python -m scripts.train_model --model xgboost"
        )
    return joblib.load(path)


def parse_tle_block(text: str) -> Tuple[str, str]:
    """Parse a 2-line TLE block from textarea text, with basic validation."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("TLE must contain at least 2 non-empty lines.")
    l1, l2 = lines[0], lines[1]
    if not (l1.startswith("1 ") and l2.startswith("2 ")):
        raise ValueError("TLE lines must start with '1 ' and '2 '.")
    return l1, l2


def satrec_from_tle_block(text: str) -> Satrec:
    l1, l2 = parse_tle_block(text)
    try:
        sat = Satrec.twoline2rv(l1, l2)
    except Exception as e:
        raise ValueError(f"Failed to create Satrec from TLE: {e}")
    return sat


def compute_features_and_risk(
    satA: Satrec,
    satB: Satrec,
    model,
    search_window_hours: float = 24.0,
    fine_window_s: int = 600,
    fine_step_s: int = 5,
) -> PredictionResult:
    """
    Compute closest approach between satA and satB around 'now',
    build feature vector, and run collision-risk model.
    """
    # Center search around "now" in UTC
    now = datetime.now(timezone.utc)
    jd_center, fr = jday(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second + now.microsecond / 1e6,
    )
    jd_center_float = jd_center + fr

    # Refine closest approach
    tca_jd, miss_km, vrel_kms = closest_approach(
        satA,
        satB,
        jd_center_float,
        window_s=fine_window_s,
        step_s=fine_step_s,
    )

    # Position / velocity at TCA
    rA, vA = _pos_vel_from_sat(satA, tca_jd)
    rB, vB = _pos_vel_from_sat(satB, tca_jd)

    # Core geometric / kinematic features
    feats = pair_features(rA, vA, rB, vB)

    # Add the extra features used in training
    feats_ext = dict(feats)
    feats_ext["tca_jd"] = float(tca_jd)
    feats_ext["miss_km"] = float(miss_km)
    feats_ext["vrel_kms"] = float(vrel_kms)

    # Build single-row DataFrame with correct columns
    # (missing keys will raise KeyError, which is good – consistent with training)
    feature_row = {col: feats_ext[col] for col in FEATURE_COLUMNS}
    X = pd.DataFrame([feature_row])

    prob = float(model.predict_proba(X)[0, 1])
    label = int(prob >= 0.5)

    return PredictionResult(
        tca_jd=float(tca_jd),
        miss_km=float(miss_km),
        vrel_kms=float(vrel_kms),
        prob=prob,
        label=label,
        features=feature_row,
    )


def risk_band(prob: float) -> Tuple[str, str]:
    """Map probability to a human label + color."""
    if prob >= 0.7:
        return "High", "#ff4b4b"
    elif prob >= 0.3:
        return "Medium", "#faca2b"
    else:
        return "Low", "#21c55d"


def render_orbit_plot(satA: Satrec, satB: Satrec, tca_jd: float):
    """Call orbit_plot.plot_orbits_3d with flexible signature."""
    try:
        # Try the most feature-rich signature first
        try:
            fig = plot_orbits_3d(satA, satB, tca_jd)
        except TypeError:
            # Fallback: older version might only accept (satA, satB)
            fig = plot_orbits_3d(satA, satB)
        return fig
    except Exception as e:
        st.warning(f"Could not generate 3D orbit plot: {e}")
        return None


def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Global background + font */
        body {
            background-color: #05060a;
        }
        .stApp {
            background: radial-gradient(circle at top, #0f172a 0, #020617 55%);
            color: #e5e7eb;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                         "Inter", sans-serif;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #020617;
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #e5e7eb !important;
        }
        /* Cards */
        .metric-card {
            padding: 1rem 1.25rem;
            border-radius: 0.9rem;
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.7));
            border: 1px solid rgba(148,163,184,0.4);
            box-shadow: 0 20px 35px rgba(15,23,42,0.65);
        }
        .metric-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #9ca3af;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 600;
            margin-top: 0.15rem;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #9ca3af;
            margin-top: 0.35rem;
        }
        /* Buttons */
        button[kind="primary"] {
            border-radius: 999px !important;
            border: 1px solid rgba(56,189,248,0.4) !important;
            background: radial-gradient(circle at top left, #22d3ee 0, #0ea5e9 40%, #0369a1 100%) !important;
            color: #0b1120 !important;
            font-weight: 600 !important;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-size: 0.8rem !important;
        }
        button[kind="primary"]:hover {
            filter: brightness(1.1);
        }
        /* Textareas */
        textarea {
            font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
            font-size: 0.78rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Kṣatra – Collision Risk Dashboard",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # Sidebar: inputs
    with st.sidebar:
        st.markdown("### 🛰️ TLE Inputs")
        st.caption("Paste 2-line TLEs for each object.")

        tle_a_text = st.text_area(
            "Satellite A TLE",
            value=DEFAULT_TLE_A,
            height=120,
            key="tle_a",
        )
        tle_b_text = st.text_area(
            "Satellite B TLE",
            value=DEFAULT_TLE_B,
            height=120,
            key="tle_b",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Model & Analysis Options")

        threshold_km = st.slider(
            "Near-miss threshold (km)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="This does not change the trained model, but helps interpret the miss distance.",
        )

        fine_window_s = st.slider(
            "Refinement window (seconds)",
            min_value=300,
            max_value=3600,
            value=600,
            step=60,
            help="Time window around the initial guess where TCA refinement is performed.",
        )

        fine_step_s = st.select_slider(
            "Refinement step (seconds)",
            options=[1, 2, 5, 10, 20],
            value=5,
        )

        run_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    # Main layout
    st.markdown(
        """
        <h1 style="font-size: 2.6rem; margin-bottom: 0.1rem;">
            Kṣatra – Collision Risk Prediction Dashboard
        </h1>
        <p style="color:#9ca3af; max-width: 740px; font-size: 0.9rem;">
            Real-time orbital conjunction analysis powered by SGP4 propagation and an XGBoost-based
            machine-learning model trained on synthetic and real close-approach encounters.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.write("")  # small spacer

    col_metrics, col_plot = st.columns([0.45, 0.55])

    # Default placeholders
    with col_metrics:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-title">Status</div>'
            '<div class="metric-value">Awaiting input</div>'
            '<div class="metric-sub">Provide TLEs and click “Run Prediction”.</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col_plot:
        st.markdown(
            '<div class="metric-card"><div class="metric-title">'
            "3D Orbit Visualization"
            "</div><div class='metric-sub'>Orbits will appear here once a prediction is run.</div></div>",
            unsafe_allow_html=True,
        )

    # ----------------- RUN PREDICTION LOGIC -----------------
    if run_btn:
        try:
            model = load_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # Parse TLEs
        try:
            satA = satrec_from_tle_block(tle_a_text)
        except Exception as e:
            st.error(f"Satellite A TLE error: {e}")
            st.stop()

        try:
            satB = satrec_from_tle_block(tle_b_text)
        except Exception as e:
            st.error(f"Satellite B TLE error: {e}")
            st.stop()

        # Compute physics + ML
        with st.spinner("Propagating orbits and running collision prediction..."):
            try:
                result = compute_features_and_risk(
                    satA,
                    satB,
                    model,
                    fine_window_s=int(fine_window_s),
                    fine_step_s=int(fine_step_s),
                )
            except Exception as e:
                st.error(f"Failed to compute closest approach / prediction: {e}")
                st.stop()

        band, band_color = risk_band(result.prob)

        # --- Update metrics card ---
        with col_metrics:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-title">Predicted Risk Band</div>
                  <div class="metric-value" style="color:{band_color};">{band}</div>
                  <div class="metric-sub">
                    Probability of collision: <b>{result.prob:.4f}</b><br/>
                    Miss distance at TCA: <b>{result.miss_km:.3f} km</b><br/>
                    Relative speed: <b>{result.vrel_kms:.3f} km/s</b><br/>
                    Threshold for "near miss": <b>{threshold_km:.1f} km</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- Orbit plot ---
        with col_plot:
            fig = render_orbit_plot(satA, satB, result.tca_jd)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("3D orbit plot is unavailable. Check terminal logs for details.")

        # --- Detailed feature table ---
        st.markdown("### 🔎 Features Used by the Model")
        feat_df = pd.DataFrame([result.features])
        st.dataframe(
            feat_df.T.rename(columns={0: "value"}),
            use_container_width=True,
            height=320,
        )

        # Small textual summary
        st.markdown(
            f"""
            **Summary**

            - Time of closest approach (TCA, JD): `{result.tca_jd:.6f}`
            - Predicted miss distance: **{result.miss_km:.3f} km**
            - Model-estimated collision probability: **{result.prob:.4f}**  
            - Classification: **{band} risk** (label = {result.label})
            """
        )


if __name__ == "__main__":
    main()
