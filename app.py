"""
GPS & DGPS Interactive Teaching Simulator
Glass-Box Edition — every calculation is visible and explained

Run with:  streamlit run app.py

Fixes applied (v2):
  #1  Broken bold in step log  — regex paired replace
  #2  Invalid fillcolor on Box plot  — explicit rgba constants
  #3  Ref-station marker anchored to true position
  #4  Monte Carlo slow first load  — lightweight default; full run on button only
  #5  Sky-plot elevation rings  — single combined trace, no legend pollution
  #6  Dead imports removed  (time, plotly.express)
  #7  Magic number 0.001 commented
  #8  HTML helpers  — card/formula/step builders reduce duplication
  #9  Variable shadowing  — loop var renamed col_name
  #11 Stale-state warning  — info banner when sliders changed but not run
  #12 Sky-plot annotation positions  — relative to chart, not hardcoded paper fractions
  #13 Safe correction lookup  — .get() with fallback prevents KeyError
"""

import re

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gps_core import GPSSimulator, monte_carlo_simulation, ReceiverPosition

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPS & DGPS Simulator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main { background: #0a0e1a; }
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
}
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 12px 0;
    backdrop-filter: blur(10px);
}
.metric-gps {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 12px; padding: 16px; text-align: center;
}
.metric-dgps {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(52,211,153,0.05));
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 12px; padding: 16px; text-align: center;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
    color: #94a3b8; margin-bottom: 6px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px; font-weight: 700; color: #f1f5f9;
}
.formula-box {
    background: rgba(15,23,42,0.8);
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 13px; color: #7dd3fc; margin: 8px 0;
}
.step-box {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 8px; padding: 10px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 12px; color: #cbd5e1; margin: 4px 0;
}
.sat-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px; padding: 12px; margin: 4px 0;
    font-family: 'Space Mono', monospace; font-size: 12px;
}
.concept-header {
    font-family: 'Syne', sans-serif;
    font-size: 13px; font-weight: 600; letter-spacing: 3px;
    text-transform: uppercase; color: #38bdf8; margin-bottom: 8px;
}
.stSlider > div > div { background: #1e293b !important; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03); border-radius: 10px; padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─── FIX #8: HTML helper functions ────────────────────────────────────────────
def glass_card(content: str, extra_style: str = "") -> str:
    return f'<div class="glass-card" style="{extra_style}">{content}</div>'

def formula_box(content: str) -> str:
    return f'<div class="formula-box">{content}</div>'

def step_box(content: str) -> str:
    # FIX #1: Regex paired **text** → <b>text</b> (old code broke on second replace)
    safe = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)
    return f'<div class="step-box">{safe}</div>'

def sat_card(content: str, border_color: str = "rgba(56,189,248,0.2)") -> str:
    return f'<div class="sat-card" style="border-color:{border_color};">{content}</div>'

def concept_header(text: str) -> str:
    return f'<div class="concept-header">{text}</div>'


# ─── Shared plot theme ─────────────────────────────────────────────────────────
PLOT_THEME = dict(
    plot_bgcolor="#0d1526",
    paper_bgcolor="#0d1526",
    font=dict(color="#cbd5e1", family="Space Mono"),
)
GRID = dict(gridcolor="rgba(148,163,184,0.1)")
LEGEND_STYLE = dict(bgcolor="rgba(13,21,38,0.8)",
                    bordercolor="rgba(99,179,237,0.3)", borderwidth=1)

# Explicit rgba for Box plot fill — FIX #2
_BOX_FILL = {"GPS Error (m)": "rgba(248,113,113,0.2)",
             "DGPS Error (m)": "rgba(52,211,153,0.2)"}


# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ Simulation Controls")
    st.markdown("---")

    st.markdown("### 📡 Constellation")
    n_satellites = st.slider("Number of Satellites", 4, 12, 8,
                              help="Minimum 4 needed for 3D fix")

    st.markdown("### ⚠️ Error Sources")
    st.markdown(concept_header("Atmospheric"), unsafe_allow_html=True)
    iono_scale  = st.slider("Ionospheric Scale", 0.0, 3.0, 1.0, 0.1,
                             help="~7 m zenith delay × elevation mapping function")
    tropo_scale = st.slider("Tropospheric Scale", 0.0, 2.0, 1.0, 0.1,
                             help="~2.3 m zenith delay × mapping function")

    st.markdown(concept_header("Local"), unsafe_allow_html=True)
    multi_scale = st.slider("Multipath Scale", 0.0, 3.0, 1.0, 0.1,
                             help="Signal reflections from buildings / ground")
    clock_scale = st.slider("Clock Error Scale", 0.0, 3.0, 1.0, 0.1,
                             help="Satellite clock bias — removable by DGPS")

    st.markdown("### 🏗️ DGPS Setup")
    ref_dist_km = st.slider("Reference Station Distance (km)", 1, 100, 10,
                             help="Closer station → better error correlation")

    st.markdown("### 🎲 Simulation")
    seed   = st.number_input("Random Seed", 0, 9999, 42,
                              help="Different seeds → different satellite geometries")
    n_runs = st.slider("Monte Carlo Runs", 50, 500, 200, 50,
                        help="Larger = more accurate statistics, slower")

    run_btn = st.button("▶  Run Simulation", use_container_width=True, type="primary")
    mc_btn  = st.button("📊  Run Monte Carlo", use_container_width=True)


# ─── Session-state keys that capture the last-run parameters ──────────────────
_PARAM_KEYS = ("n_satellites", "iono_scale", "tropo_scale",
               "multi_scale", "clock_scale", "ref_dist_km", "seed")

def _current_params() -> tuple:
    return (n_satellites, iono_scale, tropo_scale,
            multi_scale, clock_scale, ref_dist_km, seed)


# ─── Initialise / refresh simulation result ───────────────────────────────────
if "result" not in st.session_state or run_btn:
    sim = GPSSimulator(seed=int(seed))
    st.session_state.result = sim.run(
        n_satellites=n_satellites,
        ionospheric_scale=iono_scale,
        tropospheric_scale=tropo_scale,
        multipath_scale=multi_scale,
        clock_scale=clock_scale,
        ref_station_offset_km=ref_dist_km,
    )
    st.session_state.last_params = _current_params()

# FIX #4: Lightweight default MC on first load; full run only on button press
if "mc_df" not in st.session_state:
    st.session_state.mc_df = monte_carlo_simulation(
        n_runs=50,                  # fast default — user sees something immediately
        n_satellites=n_satellites,
        ionospheric_scale=iono_scale,
        tropospheric_scale=tropo_scale,
        multipath_scale=multi_scale,
    )
if mc_btn:
    with st.spinner(f"Running {n_runs} Monte Carlo trials…"):
        st.session_state.mc_df = monte_carlo_simulation(
            n_runs=n_runs,
            n_satellites=n_satellites,
            ionospheric_scale=iono_scale,
            tropospheric_scale=tropo_scale,
            multipath_scale=multi_scale,
        )

result = st.session_state.result
mc_df  = st.session_state.mc_df


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:24px 0 12px;">
  <h1 style="font-size:2.6rem; color:#f1f5f9; margin:0; letter-spacing:-1px;">
    🛰️ GPS &amp; DGPS Teaching Simulator
  </h1>
  <p style="color:#64748b; font-family:'Space Mono',monospace;
            font-size:13px; letter-spacing:2px; margin-top:8px;">
    GLASS-BOX EDITION — EVERY CALCULATION IS VISIBLE
  </p>
</div>
""", unsafe_allow_html=True)

# FIX #11: Stale-state warning when sliders changed but simulation not re-run
if st.session_state.get("last_params") != _current_params():
    st.info("⚙️ Parameters changed — press **▶ Run Simulation** to update results.",
            icon="ℹ️")


# ─── Top Metrics Row ──────────────────────────────────────────────────────────
gps_err  = result.true_position.distance_to(result.gps_position)
dgps_err = result.true_position.distance_to(result.dgps_position)
# FIX #7: guard against near-zero GPS error (prevents division-by-zero)
improvement = (1 - dgps_err / max(gps_err, 0.001)) * 100

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        f'<div class="metric-gps"><div class="metric-label">GPS Error</div>'
        f'<div class="metric-value" style="color:#f87171;">{gps_err:.1f} m</div></div>',
        unsafe_allow_html=True)
with col2:
    st.markdown(
        f'<div class="metric-dgps"><div class="metric-label">DGPS Error</div>'
        f'<div class="metric-value" style="color:#34d399;">{dgps_err:.2f} m</div></div>',
        unsafe_allow_html=True)
with col3:
    st.markdown(
        f'<div class="glass-card" style="text-align:center;padding:16px;">'
        f'<div class="metric-label">Improvement</div>'
        f'<div class="metric-value" style="color:#38bdf8;">{improvement:.0f}%</div></div>',
        unsafe_allow_html=True)
with col4:
    hdop = result.dop_values["HDOP"]
    dop_color = "#34d399" if hdop < 2 else "#fbbf24" if hdop < 4 else "#f87171"
    st.markdown(
        f'<div class="glass-card" style="text-align:center;padding:16px;">'
        f'<div class="metric-label">HDOP</div>'
        f'<div class="metric-value" style="color:{dop_color};">{hdop:.2f}</div></div>',
        unsafe_allow_html=True)
with col5:
    st.markdown(
        f'<div class="glass-card" style="text-align:center;padding:16px;">'
        f'<div class="metric-label">Satellites</div>'
        f'<div class="metric-value" style="color:#a78bfa;">{n_satellites}</div></div>',
        unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🗺️  Position Map",
    "📡  Sky Plot & DOP",
    "⚠️  Error Breakdown",
    "🔧  DGPS Corrections",
    "📊  Statistics",
    "📚  How It Works",
    "🔬  Step-by-Step Math",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — POSITION MAP
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Position Fix Comparison")
    col_map, col_info = st.columns([3, 1])

    with col_map:
        fig = go.Figure()

        # True position
        fig.add_trace(go.Scatter(
            x=[result.true_position.x_m], y=[result.true_position.y_m],
            mode="markers", name="True Position",
            marker=dict(symbol="cross", size=20, color="#f1f5f9",
                        line=dict(width=3, color="#f1f5f9")),
        ))
        # GPS fix
        fig.add_trace(go.Scatter(
            x=[result.gps_position.x_m], y=[result.gps_position.y_m],
            mode="markers", name=f"GPS Fix ({gps_err:.1f} m error)",
            marker=dict(symbol="circle", size=14, color="#f87171",
                        line=dict(width=2, color="#fca5a5")),
        ))
        # DGPS fix
        fig.add_trace(go.Scatter(
            x=[result.dgps_position.x_m], y=[result.dgps_position.y_m],
            mode="markers", name=f"DGPS Fix ({dgps_err:.2f} m error)",
            marker=dict(symbol="circle", size=14, color="#34d399",
                        line=dict(width=2, color="#6ee7b7")),
        ))
        # Error lines (true → fix)
        for pos, color in [(result.gps_position, "#f87171"),
                           (result.dgps_position, "#34d399")]:
            fig.add_trace(go.Scatter(
                x=[result.true_position.x_m, pos.x_m],
                y=[result.true_position.y_m, pos.y_m],
                mode="lines", showlegend=False,
                line=dict(color=color, width=1.5, dash="dot"),
            ))
        # Error circles (CEP)
        theta = np.linspace(0, 2 * np.pi, 100)
        for r, fill, line_c, name in [
            (gps_err,  "rgba(248,113,113,0.12)", "rgba(248,113,113,0.7)", f"GPS {gps_err:.1f} m CEP"),
            (dgps_err, "rgba(52,211,153,0.12)",  "rgba(52,211,153,0.7)",  f"DGPS {dgps_err:.2f} m CEP"),
        ]:
            fig.add_trace(go.Scatter(
                x=result.true_position.x_m + r * np.cos(theta),
                y=result.true_position.y_m + r * np.sin(theta),
                mode="lines", name=name,
                line=dict(color=line_c, width=1, dash="dash"),
                fill="toself", fillcolor=fill,
            ))

        # FIX #3: Reference station position anchored to true position
        ref_x = result.true_position.x_m + ref_dist_km * 1000
        ref_y = result.true_position.y_m
        fig.add_trace(go.Scatter(
            x=[ref_x], y=[ref_y],
            mode="markers+text", name="DGPS Reference Station",
            text=["REF"], textposition="top center",
            textfont=dict(color="#38bdf8", size=11),
            marker=dict(symbol="diamond", size=14, color="#38bdf8",
                        line=dict(width=2, color="#7dd3fc")),
        ))

        fig.update_layout(
            **PLOT_THEME,
            legend=dict(**LEGEND_STYLE, font=dict(size=11)),
            xaxis=dict(title="East (m)", zeroline=False, showgrid=True, **GRID),
            yaxis=dict(title="North (m)", zeroline=False, showgrid=True,
                       scaleanchor="x", **GRID),
            height=500, margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("**Position Summary**")
        st.markdown(
            sat_card(
                f'<span style="color:#94a3b8;">True</span><br>'
                f'x = {result.true_position.x_m:.2f} m<br>'
                f'y = {result.true_position.y_m:.2f} m'
            ) +
            sat_card(
                f'<span style="color:#f87171;">GPS Fix</span><br>'
                f'x = {result.gps_position.x_m:.2f} m<br>'
                f'y = {result.gps_position.y_m:.2f} m<br>'
                f'Δ = <b style="color:#f87171;">{gps_err:.2f} m</b>',
                border_color="rgba(248,113,113,0.4)"
            ) +
            sat_card(
                f'<span style="color:#34d399;">DGPS Fix</span><br>'
                f'x = {result.dgps_position.x_m:.2f} m<br>'
                f'y = {result.dgps_position.y_m:.2f} m<br>'
                f'Δ = <b style="color:#34d399;">{dgps_err:.2f} m</b>',
                border_color="rgba(52,211,153,0.4)"
            ),
            unsafe_allow_html=True,
        )

        st.markdown("**DOP Values**")
        for k, v in result.dop_values.items():
            color = "#34d399" if v < 2 else "#fbbf24" if v < 4 else "#f87171"
            st.markdown(
                step_box(f'{k}: <span style="color:{color};font-weight:700;">{v:.2f}</span>'),
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SKY PLOT & DOP
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    col_sky, col_dop = st.columns([1, 1])

    with col_sky:
        st.markdown("### Sky Plot")
        st.caption("Satellite positions as seen from receiver. Centre = zenith, edge = horizon.")

        fig_sky = go.Figure()

        # FIX #5: Single combined polar trace for elevation rings (no legend entries)
        ring_r, ring_theta = [], []
        for el_ring in [10, 30, 60, 90]:
            t = np.linspace(0, 360, 120)
            r_val = 1 - el_ring / 90
            ring_r.extend([r_val] * 120 + [None])
            ring_theta.extend(list(t) + [None])

        fig_sky.add_trace(go.Scatterpolar(
            r=ring_r, theta=ring_theta,
            mode="lines", showlegend=False,
            line=dict(color="rgba(148,163,184,0.2)", width=1),
        ))

        # Satellites
        for sat in result.satellites:
            r_plot = 1 - sat.elevation_deg / 90
            error_mag = abs(sat.total_error_m)
            red   = min(255, int(error_mag * 15))
            green = max(0, 255 - int(error_mag * 15))
            color = f"rgba({red},{green},100,0.9)"
            fig_sky.add_trace(go.Scatterpolar(
                r=[r_plot], theta=[sat.azimuth_deg],
                mode="markers+text",
                text=[f"G{sat.prn:02d}"],
                textposition="top center",
                textfont=dict(size=9, color="#cbd5e1"),
                marker=dict(size=18, color=color, line=dict(color="white", width=1)),
                name=f"PRN {sat.prn} ({sat.elevation_deg:.0f}°)",
                showlegend=False,
            ))

        # FIX #12: Elevation labels as polar annotations (not hardcoded paper fractions)
        for el_ring in [10, 30, 60]:
            r_val = 1 - el_ring / 90
            fig_sky.add_trace(go.Scatterpolar(
                r=[r_val], theta=[22],      # fixed azimuth so they don't overlap sats
                mode="text",
                text=[f"{el_ring}°"],
                textfont=dict(size=8, color="#64748b"),
                showlegend=False,
            ))

        fig_sky.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=False,
                                gridcolor="rgba(148,163,184,0.15)"),
                angularaxis=dict(
                    tickmode="array",
                    tickvals=[0, 90, 180, 270],
                    ticktext=["N", "E", "S", "W"],
                    direction="clockwise",
                    gridcolor="rgba(148,163,184,0.15)",
                    tickfont=dict(color="#94a3b8"),
                ),
                bgcolor="#0d1526",
            ),
            paper_bgcolor="#0d1526",
            font=dict(color="#cbd5e1"),
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_sky, use_container_width=True)
        st.caption("🔴 High error  🟢 Low error  (colour = total pseudorange error)")

    with col_dop:
        st.markdown("### DOP Gauges")
        st.caption("Dilution of Precision — geometric quality of the satellite spread")

        dop_info = {
            "GDOP": ("Geometric",    "#a78bfa"),
            "PDOP": ("Position 3D",  "#38bdf8"),
            "HDOP": ("Horizontal",   "#34d399"),
            "VDOP": ("Vertical",     "#fbbf24"),
            "TDOP": ("Time",         "#fb923c"),
        }
        fig_dop = go.Figure()
        for key, (label, color) in dop_info.items():
            val = min(result.dop_values[key], 10)
            fig_dop.add_trace(go.Bar(
                x=[val], y=[f"{key}\n{label}"],
                orientation="h",
                marker=dict(color=color, opacity=0.8, line=dict(color=color, width=1)),
                name=key,
                text=[f"{result.dop_values[key]:.2f}"],
                textposition="outside",
                textfont=dict(color="#f1f5f9"),
                showlegend=False,
            ))
        for x_val, label, color in [(1, "Ideal", "#34d399"), (2, "Excellent", "#86efac"),
                                     (5, "Good",  "#fbbf24"), (10, "Poor",    "#f87171")]:
            fig_dop.add_vline(x=x_val, line_dash="dot", line_color=color,
                               annotation_text=label,
                               annotation_font_color=color, annotation_font_size=10)
        fig_dop.update_layout(
            **PLOT_THEME,
            xaxis=dict(range=[0, 10], title="DOP Value", **GRID),
            yaxis=GRID,
            height=300, margin=dict(l=20, r=80, t=20, b=20),
        )
        st.plotly_chart(fig_dop, use_container_width=True)

        st.markdown(glass_card(
            concept_header("DOP Interpretation") +
            '<div style="font-family:\'Space Mono\',monospace;font-size:12px;'
            'color:#94a3b8;line-height:1.9;">'
            '&lt; 1 &nbsp;→ Ideal<br>1–2 &nbsp;→ Excellent<br>'
            '2–5 &nbsp;→ Good (typical)<br>5–10 → Moderate<br>'
            '&gt; 10 → Poor</div>'
        ), unsafe_allow_html=True)

        st.markdown(formula_box(
            "DOP = σ_position / σ_pseudorange<br><br>"
            "Low DOP  = satellites well spread in sky<br>"
            "High DOP = satellites clustered together"
        ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ERROR BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Per-Satellite Error Breakdown")
    st.caption("GPS error budget decomposed by source for each satellite")

    error_types = ["Satellite Clock", "Ionospheric Delay", "Tropospheric Delay",
                   "Multipath", "Receiver Noise"]
    err_colors  = ["#a78bfa", "#38bdf8", "#34d399", "#fbbf24", "#fb923c"]

    sat_labels = [f"G{s.prn:02d}" for s in result.satellites]
    fig_err = go.Figure()
    for err_type, color in zip(error_types, err_colors):
        values = [abs(sat.error_breakdown()[err_type]) for sat in result.satellites]
        fig_err.add_trace(go.Bar(name=err_type, x=sat_labels, y=values,
                                  marker_color=color, opacity=0.85))
    fig_err.update_layout(
        barmode="stack", **PLOT_THEME, legend=dict(**LEGEND_STYLE),
        xaxis=dict(title="Satellite PRN", **GRID),
        yaxis=dict(title="Error (m)", **GRID),
        height=380, margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_err, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Elevation vs Total Error** (atmospheric mapping effect)")
        fig_el = go.Figure()
        for sat in result.satellites:
            fig_el.add_trace(go.Scatter(
                x=[sat.elevation_deg], y=[abs(sat.total_error_m)],
                mode="markers+text",
                text=[f"G{sat.prn}"],
                textposition="top center",
                textfont=dict(size=9, color="#94a3b8"),
                marker=dict(size=12, color="#38bdf8", opacity=0.8),
                showlegend=False,
            ))
        el_range = np.linspace(10, 85, 50)
        nominal  = (7 * iono_scale + 2.3 * tropo_scale) / np.sin(np.radians(el_range))
        fig_el.add_trace(go.Scatter(
            x=el_range, y=nominal, mode="lines",
            name="Atm. mapping curve",
            line=dict(color="#fbbf24", dash="dash", width=1.5),
        ))
        fig_el.update_layout(
            **PLOT_THEME,
            xaxis=dict(title="Elevation (°)", **GRID),
            yaxis=dict(title="Total Error (m)", **GRID),
            height=300, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_el, use_container_width=True)

    with col_b:
        st.markdown("**Error Source Pie** (average across all satellites)")
        avg_errors: dict[str, float] = {k: 0.0 for k in error_types}
        for sat in result.satellites:
            for k, v in sat.error_breakdown().items():
                avg_errors[k] += abs(v)
        n = len(result.satellites)
        avg_errors = {k: v / n for k, v in avg_errors.items()}

        fig_pie = go.Figure(go.Pie(
            labels=list(avg_errors.keys()),
            values=list(avg_errors.values()),
            marker=dict(colors=err_colors, line=dict(color="#0d1526", width=2)),
            hole=0.4,
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0d1526",
            font=dict(color="#cbd5e1", family="Space Mono"),
            legend=dict(bgcolor="rgba(13,21,38,0.8)", font=dict(size=10)),
            height=300, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DGPS CORRECTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### DGPS Correction Vectors")
    st.caption("How the reference station isolates and broadcasts pseudorange corrections")

    col_corr, col_table = st.columns([2, 1])

    with col_corr:
        prns        = [f"G{s.prn:02d}" for s in result.satellites]
        corrections = result.dgps_correction_vector

        fig_corr = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Pseudorange Error Before / After Correction",
                             "Correction Magnitude per Satellite"],
            vertical_spacing=0.18,
        )
        before = [abs(s.total_error_m) for s in result.satellites]
        after  = [abs(s.multipath_error_m + s.receiver_noise_m)
                  for s in result.satellites]

        fig_corr.add_trace(go.Bar(x=prns, y=before, name="GPS (no correction)",
                                   marker_color="#f87171", opacity=0.8), row=1, col=1)
        fig_corr.add_trace(go.Bar(x=prns, y=after, name="After DGPS correction",
                                   marker_color="#34d399", opacity=0.8), row=1, col=1)

        # FIX #13: Safe .get() on correction dict to prevent KeyError
        corr_vals = [abs(corrections.get(s.prn, {}).get("correction_m", 0.0))
                     for s in result.satellites]
        fig_corr.add_trace(go.Bar(x=prns, y=corr_vals, name="Correction magnitude",
                                   marker_color="#38bdf8", opacity=0.8), row=2, col=1)

        fig_corr.update_layout(
            **PLOT_THEME,
            barmode="group",
            legend=dict(**LEGEND_STYLE),
            height=480, margin=dict(l=20, r=20, t=40, b=20),
        )
        for row in (1, 2):
            fig_corr.update_xaxes(**GRID, row=row, col=1)
            fig_corr.update_yaxes(title_text="Metres", **GRID, row=row, col=1)

        st.plotly_chart(fig_corr, use_container_width=True)

    with col_table:
        st.markdown("**Correction Detail**")
        rows = []
        for sat in result.satellites:
            # FIX #13: safe lookup
            c = corrections.get(sat.prn, {"correction_m": float("nan")})
            rows.append({
                "PRN":            f"G{sat.prn:02d}",
                "Total Err (m)":  f"{sat.total_error_m:.2f}",
                "Correction (m)": f"{c['correction_m']:.2f}",
                "Residual (m)":   f"{sat.multipath_error_m + sat.receiver_noise_m:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

        st.markdown(formula_box(
            "DGPS Correction:<br><br>"
            "Δρ = ρ_measured − ρ_true<br><br>"
            "Broadcast to rover.<br>"
            "Rover applies:<br>"
            "ρ_corrected = ρ_rover − Δρ"
        ), unsafe_allow_html=True)

        st.markdown(glass_card(
            '<div style="font-size:12px;font-family:\'Space Mono\',monospace;color:#94a3b8;">'
            '✅ Removes: Clock, Iono, Tropo<br>'
            '❌ Can\'t remove: Multipath, Noise<br>'
            '⚠️ Accuracy degrades with distance</div>'
        ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    n_mc = len(mc_df)
    st.markdown(f"### Monte Carlo Analysis — {n_mc} Simulation Runs")
    if n_mc == 50:
        st.caption("Showing quick 50-run default. Press **📊 Run Monte Carlo** for the full analysis.")

    col_hist, col_box = st.columns(2)

    # FIX #9: renamed loop variable from `col` to `col_name` to avoid shadowing
    with col_hist:
        st.markdown("**Error Distribution**")
        fig_hist = go.Figure()
        for col_name, color, name in [
            ("GPS Error (m)",  "#f87171", "GPS"),
            ("DGPS Error (m)", "#34d399", "DGPS"),
        ]:
            fig_hist.add_trace(go.Histogram(
                x=mc_df[col_name], name=name, opacity=0.75,
                marker_color=color, nbinsx=30,
            ))
        fig_hist.update_layout(
            barmode="overlay", **PLOT_THEME,
            legend=dict(**LEGEND_STYLE),
            xaxis=dict(title="Position Error (m)", **GRID),
            yaxis=dict(title="Count", **GRID),
            height=340, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        st.markdown("**Box Plot Comparison**")
        fig_box = go.Figure()
        for col_name, color, name in [
            ("GPS Error (m)",  "#f87171", "GPS"),
            ("DGPS Error (m)", "#34d399", "DGPS"),
        ]:
            # FIX #2: explicit rgba fillcolor — no fragile string manipulation
            fig_box.add_trace(go.Box(
                y=mc_df[col_name], name=name,
                marker_color=color,
                line_color=color,
                fillcolor=_BOX_FILL[col_name],
            ))
        fig_box.update_layout(
            **PLOT_THEME,
            yaxis=dict(title="Error (m)", **GRID),
            height=340, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("**Error Over Simulation Runs**")
    fig_ts = go.Figure()
    for col_name, color, name in [
        ("GPS Error (m)",  "#f87171", "GPS"),
        ("DGPS Error (m)", "#34d399", "DGPS"),
    ]:
        fig_ts.add_trace(go.Scatter(
            x=mc_df["run"], y=mc_df[col_name].rolling(10, min_periods=1).mean(),
            name=f"{name} (10-run avg)", line=dict(color=color, width=2),
        ))
        fig_ts.add_trace(go.Scatter(
            x=mc_df["run"], y=mc_df[col_name],
            mode="lines", line=dict(color=color, width=0.5),
            opacity=0.3, showlegend=False,
        ))
    fig_ts.update_layout(
        **PLOT_THEME, legend=dict(**LEGEND_STYLE),
        xaxis=dict(title="Run #", **GRID),
        yaxis=dict(title="Error (m)", **GRID),
        height=280, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("**Summary Statistics**")
    st.dataframe(
        mc_df[["GPS Error (m)", "DGPS Error (m)"]].describe().round(3),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    col_gps, col_dgps = st.columns(2)

    with col_gps:
        st.markdown("## 📡 How GPS Works")
        st.markdown(glass_card(
            concept_header("1. Trilateration") +
            "GPS measures signal travel time from multiple satellites. "
            "Distance = time × <i>c</i>. With 4+ satellites the receiver "
            "solves for x, y, z, and clock bias simultaneously."
        ), unsafe_allow_html=True)

        st.markdown(formula_box(
            "Pseudorange equation:<br>"
            "ρᵢ = |Xᵢ − X| + c·δt + εᵢ<br><br>"
            "ρᵢ = measured pseudorange to sat i<br>"
            "Xᵢ = satellite position (known)<br>"
            "X  = receiver position (unknown)<br>"
            "δt = receiver clock offset<br>"
            "εᵢ = all errors combined"
        ), unsafe_allow_html=True)

        st.markdown(glass_card(
            concept_header("2. Error Sources") +
            '<table style="font-family:\'Space Mono\',monospace;font-size:12px;'
            'width:100%;color:#94a3b8;">'
            '<tr><td style="color:#a78bfa;">Clock errors</td><td>~2–5 m</td><td>Satellite timing drift</td></tr>'
            '<tr><td style="color:#38bdf8;">Ionosphere</td><td>~5–15 m</td><td>Free electrons delay signal</td></tr>'
            '<tr><td style="color:#34d399;">Troposphere</td><td>~0.5–3 m</td><td>Water vapour, pressure</td></tr>'
            '<tr><td style="color:#fbbf24;">Multipath</td><td>~0–5 m</td><td>Signal reflections</td></tr>'
            '<tr><td style="color:#fb923c;">Receiver noise</td><td>~0.3 m</td><td>Thermal noise in chip</td></tr>'
            '</table>'
        ), unsafe_allow_html=True)

        st.markdown(glass_card(concept_header("3. Least-Squares Solution") +
            "The receiver linearises the equations and iteratively solves:"),
            unsafe_allow_html=True)
        st.markdown(formula_box(
            "Δx = (AᵀA)⁻¹ Aᵀ Δρ<br><br>"
            "A  = design matrix (direction cosines)<br>"
            "Δρ = pseudorange residuals<br>"
            "Δx = position correction vector<br><br>"
            "Repeat until |Δx| &lt; threshold"
        ), unsafe_allow_html=True)

    with col_dgps:
        st.markdown("## 🛰️ How DGPS Works")
        st.markdown(glass_card(
            concept_header("Key Insight") +
            "A reference station at a <b>precisely known position</b> computes "
            "the pseudorange error for every satellite and broadcasts corrections. "
            "Because atmospheric errors are spatially correlated, nearby rovers "
            "benefit dramatically."
        ), unsafe_allow_html=True)

        st.markdown(formula_box(
            "At reference station:<br>"
            "Δρᵢ = ρᵢ_measured − ρᵢ_true<br>"
            "Δρᵢ ≈ clock_err + iono + tropo<br><br>"
            "At rover:<br>"
            "ρᵢ_corrected = ρᵢ_rover − Δρᵢ<br><br>"
            "Remaining: multipath + noise only"
        ), unsafe_allow_html=True)

        st.markdown(glass_card(
            concept_header("What DGPS Removes") +
            '<table style="font-family:\'Space Mono\',monospace;font-size:12px;'
            'width:100%;color:#94a3b8;">'
            '<tr><td style="color:#34d399;">✅ Satellite clocks</td><td>Completely removed</td></tr>'
            '<tr><td style="color:#34d399;">✅ Ionospheric delay</td><td>~90% (if &lt;50 km)</td></tr>'
            '<tr><td style="color:#34d399;">✅ Tropospheric delay</td><td>~80% removed</td></tr>'
            '<tr><td style="color:#f87171;">❌ Multipath</td><td>Site-specific, uncorrelated</td></tr>'
            '<tr><td style="color:#f87171;">❌ Receiver noise</td><td>Independent at each site</td></tr>'
            '</table>'
        ), unsafe_allow_html=True)

        st.markdown(glass_card(
            concept_header("DGPS vs RTK vs PPP") +
            '<table style="font-family:\'Space Mono\',monospace;font-size:12px;'
            'width:100%;color:#94a3b8;">'
            '<tr><th style="color:#f1f5f9;">System</th>'
            '<th style="color:#f1f5f9;">Accuracy</th>'
            '<th style="color:#f1f5f9;">Range</th></tr>'
            '<tr><td>GPS alone</td><td>3–10 m</td><td>Global</td></tr>'
            '<tr><td style="color:#38bdf8;">DGPS</td><td>0.5–3 m</td><td>&lt;300 km</td></tr>'
            '<tr><td style="color:#a78bfa;">RTK</td><td>1–3 cm</td><td>&lt;30 km</td></tr>'
            '<tr><td style="color:#34d399;">PPP</td><td>2–10 cm</td><td>Global</td></tr>'
            '</table>'
        ), unsafe_allow_html=True)

        st.markdown(glass_card(
            concept_header("Real-World Applications") +
            '<div style="font-family:\'Space Mono\',monospace;font-size:12px;'
            'color:#94a3b8;line-height:2.0;">'
            '🚢 Maritime navigation (USCG beacons)<br>'
            '✈️ Aircraft approach guidance (GBAS)<br>'
            '🚜 Precision agriculture (&lt;30 cm row guidance)<br>'
            '🗺️ Survey-grade mapping<br>'
            '🚗 Autonomous vehicle lane-keeping<br>'
            '⛽ Offshore platform positioning</div>'
        ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — STEP-BY-STEP MATH
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### Step-by-Step Computation Log")
    st.caption("Every iteration of the least-squares solver — fully traceable")

    for entry in result.step_log:
        if entry.startswith("##"):
            st.markdown(entry)
        elif entry == "---":
            st.markdown("---")
        else:
            # FIX #1: regex replaces paired **bold** correctly
            st.markdown(step_box(entry), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Full Satellite Data Table")
    sat_rows = []
    for sat in result.satellites:
        sat_rows.append({
            "PRN":             f"G{sat.prn:02d}",
            "El (°)":          f"{sat.elevation_deg:.1f}",
            "Az (°)":          f"{sat.azimuth_deg:.1f}",
            "True Range (m)":  f"{sat.true_range_m:.1f}",
            "Clock (m)":       f"{sat.clock_error_m:.3f}",
            "Iono (m)":        f"{sat.ionospheric_error_m:.3f}",
            "Tropo (m)":       f"{sat.tropospheric_error_m:.3f}",
            "Multipath (m)":   f"{sat.multipath_error_m:.3f}",
            "Noise (m)":       f"{sat.receiver_noise_m:.3f}",
            "Total Err (m)":   f"{sat.total_error_m:.3f}",
            "Pseudorange (m)": f"{sat.pseudorange_m:.3f}",
        })
    st.dataframe(pd.DataFrame(sat_rows), use_container_width=True)

    st.markdown("### Atmospheric Mapping Function")
    st.markdown(formula_box(
        "M(El) = 1 / sin(El)<br><br>"
        "Scales zenith delay to slant path through atmosphere.<br>"
        "El = 90° (overhead) → M = 1.00 (shortest path)<br>"
        "El = 10° (horizon)  → M = 5.76 (5.76× more atmosphere)<br><br>"
        "Iono delay  ≈ 7.0 m × iono_scale × M(El)<br>"
        "Tropo delay ≈ 2.3 m × tropo_scale × M(El)"
    ), unsafe_allow_html=True)

    el_vals  = np.arange(10, 91, 1)
    map_vals = 1 / np.sin(np.radians(el_vals))
    fig_map  = go.Figure()
    fig_map.add_trace(go.Scatter(
        x=el_vals, y=map_vals, mode="lines", name="M(El)",
        line=dict(color="#38bdf8", width=2),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.1)",
    ))
    fig_map.update_layout(
        **PLOT_THEME,
        xaxis=dict(title="Elevation (°)", **GRID),
        yaxis=dict(title="Mapping Factor M(El)", **GRID),
        height=280, margin=dict(l=20, r=20, t=10, b=20),
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:32px 0 16px;
     font-family:'Space Mono',monospace; font-size:11px;
     letter-spacing:2px; color:#334155;">
    GPS &amp; DGPS TEACHING SIMULATOR · GLASS-BOX EDITION · v2.0<br>
    All physics, all math, all visible — built for education
</div>
""", unsafe_allow_html=True)
