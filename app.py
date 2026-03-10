"""
GPS & DGPS Interactive Teaching Simulator
Glass-Box Edition — every calculation is visible and explained

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

from gps_core import GPSSimulator, monte_carlo_simulation, ReceiverPosition

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPS & DGPS Simulator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

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
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.metric-dgps {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(52,211,153,0.05));
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
}

.formula-box {
    background: rgba(15,23,42,0.8);
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #7dd3fc;
    margin: 8px 0;
}

.step-box {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #cbd5e1;
    margin: 4px 0;
}

.error-bar-gps  { color: #f87171; }
.error-bar-dgps { color: #34d399; }

.sat-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    padding: 12px;
    margin: 4px 0;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
}

.concept-header {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 8px;
}

.sidebar-section {
    border-top: 1px solid rgba(255,255,255,0.1);
    padding-top: 12px;
    margin-top: 12px;
}

/* Streamlit element overrides */
.stSlider > div > div { background: #1e293b !important; }
[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ Simulation Controls")
    st.markdown("---")

    st.markdown("### 📡 Constellation")
    n_satellites = st.slider("Number of Satellites", 4, 12, 8,
                              help="Minimum 4 needed for 3D fix")

    st.markdown("### ⚠️ Error Sources")
    st.markdown('<div class="concept-header">Atmospheric</div>', unsafe_allow_html=True)
    iono_scale  = st.slider("Ionospheric Scale", 0.0, 3.0, 1.0, 0.1,
                             help="Ionospheric delay: ~7m at zenith × elevation mapping")
    tropo_scale = st.slider("Tropospheric Scale", 0.0, 2.0, 1.0, 0.1,
                             help="Tropospheric delay: ~2.3m at zenith × mapping function")

    st.markdown('<div class="concept-header">Local</div>', unsafe_allow_html=True)
    multi_scale = st.slider("Multipath Scale", 0.0, 3.0, 1.0, 0.1,
                             help="Signal reflections from buildings/ground")
    clock_scale = st.slider("Clock Error Scale", 0.0, 3.0, 1.0, 0.1,
                             help="Satellite clock bias (removable by DGPS)")

    st.markdown("### 🏗️ DGPS Setup")
    ref_dist_km = st.slider("Reference Station Distance (km)", 1, 100, 10,
                             help="Closer = better correlation of errors")

    st.markdown("### 🎲 Simulation")
    seed = st.number_input("Random Seed", 0, 9999, 42,
                            help="Change for different satellite geometries")
    n_runs = st.slider("Monte Carlo Runs", 50, 500, 200, 50,
                        help="Statistical analysis sample size")

    run_btn = st.button("▶  Run Simulation", use_container_width=True, type="primary")
    mc_btn  = st.button("📊  Run Monte Carlo", use_container_width=True)


# ─── Initialize session state ─────────────────────────────────────────────────
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

if "mc_df" not in st.session_state or mc_btn:
    with st.spinner("Running Monte Carlo simulation…"):
        st.session_state.mc_df = monte_carlo_simulation(
            n_runs=n_runs,
            n_satellites=n_satellites,
            ionospheric_scale=iono_scale,
            tropospheric_scale=tropo_scale,
            multipath_scale=multi_scale,
        )

result = st.session_state.result
mc_df  = st.session_state.mc_df


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 24px 0 12px;">
    <h1 style="font-size:2.6rem; color:#f1f5f9; margin:0; letter-spacing:-1px;">
        🛰️ GPS & DGPS Teaching Simulator
    </h1>
    <p style="color:#64748b; font-family:'Space Mono',monospace; font-size:13px; letter-spacing:2px; margin-top:8px;">
        GLASS-BOX EDITION — EVERY CALCULATION IS VISIBLE
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Top Metrics Row ─────────────────────────────────────────────────────────
gps_err  = result.true_position.distance_to(result.gps_position)
dgps_err = result.true_position.distance_to(result.dgps_position)
improvement = (1 - dgps_err / max(gps_err, 0.001)) * 100

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="metric-gps">
        <div class="metric-label">GPS Error</div>
        <div class="metric-value" style="color:#f87171;">{gps_err:.1f} m</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-dgps">
        <div class="metric-label">DGPS Error</div>
        <div class="metric-value" style="color:#34d399;">{dgps_err:.2f} m</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="glass-card" style="text-align:center;padding:16px;">
        <div class="metric-label">Improvement</div>
        <div class="metric-value" style="color:#38bdf8;">{improvement:.0f}%</div>
    </div>""", unsafe_allow_html=True)
with col4:
    hdop = result.dop_values['HDOP']
    dop_color = "#34d399" if hdop < 2 else "#fbbf24" if hdop < 4 else "#f87171"
    st.markdown(f"""
    <div class="glass-card" style="text-align:center;padding:16px;">
        <div class="metric-label">HDOP</div>
        <div class="metric-value" style="color:{dop_color};">{hdop:.2f}</div>
    </div>""", unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div class="glass-card" style="text-align:center;padding:16px;">
        <div class="metric-label">Satellites</div>
        <div class="metric-value" style="color:#a78bfa;">{n_satellites}</div>
    </div>""", unsafe_allow_html=True)


# ─── Main Tab Layout ─────────────────────────────────────────────────────────
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
# TAB 1: POSITION MAP
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Position Fix Comparison")

    col_map, col_info = st.columns([3, 1])

    with col_map:
        fig = go.Figure()

        # True position
        fig.add_trace(go.Scatter(
            x=[result.true_position.x_m],
            y=[result.true_position.y_m],
            mode="markers",
            name="True Position",
            marker=dict(symbol="cross", size=20, color="#f1f5f9",
                        line=dict(width=3, color="#f1f5f9")),
        ))

        # GPS position
        fig.add_trace(go.Scatter(
            x=[result.gps_position.x_m],
            y=[result.gps_position.y_m],
            mode="markers",
            name=f"GPS Fix ({gps_err:.1f} m error)",
            marker=dict(symbol="circle", size=14, color="#f87171",
                        line=dict(width=2, color="#fca5a5")),
        ))

        # DGPS position
        fig.add_trace(go.Scatter(
            x=[result.dgps_position.x_m],
            y=[result.dgps_position.y_m],
            mode="markers",
            name=f"DGPS Fix ({dgps_err:.2f} m error)",
            marker=dict(symbol="circle", size=14, color="#34d399",
                        line=dict(width=2, color="#6ee7b7")),
        ))

        # Error lines
        for pos, color, dash in [
            (result.gps_position, "#f87171", "dot"),
            (result.dgps_position, "#34d399", "dot"),
        ]:
            fig.add_trace(go.Scatter(
                x=[result.true_position.x_m, pos.x_m],
                y=[result.true_position.y_m, pos.y_m],
                mode="lines",
                showlegend=False,
                line=dict(color=color, width=1.5, dash=dash),
            ))

        # Error circles
        theta = np.linspace(0, 2*np.pi, 100)
        for r, color, name in [
            (gps_err, "rgba(248,113,113,0.15)", f"GPS {gps_err:.1f}m CEP"),
            (dgps_err, "rgba(52,211,153,0.15)", f"DGPS {dgps_err:.2f}m CEP"),
        ]:
            fig.add_trace(go.Scatter(
                x=result.true_position.x_m + r * np.cos(theta),
                y=result.true_position.y_m + r * np.sin(theta),
                mode="lines", name=name,
                line=dict(color=color.replace("0.15", "0.6"), width=1, dash="dash"),
                fill="toself", fillcolor=color,
            ))

        # Reference station
        ref_x = ref_dist_km * 1000
        fig.add_trace(go.Scatter(
            x=[ref_x],
            y=[0],
            mode="markers+text",
            name="DGPS Reference Station",
            text=["REF"],
            textposition="top center",
            textfont=dict(color="#38bdf8", size=11),
            marker=dict(symbol="diamond", size=14, color="#38bdf8",
                        line=dict(width=2, color="#7dd3fc")),
        ))

        fig.update_layout(
            plot_bgcolor="#0d1526",
            paper_bgcolor="#0d1526",
            font=dict(color="#cbd5e1", family="Space Mono"),
            legend=dict(
                bgcolor="rgba(13,21,38,0.8)",
                bordercolor="rgba(99,179,237,0.3)",
                borderwidth=1,
                font=dict(size=11),
            ),
            xaxis=dict(
                title="East (m)", gridcolor="rgba(148,163,184,0.1)",
                zeroline=False, showgrid=True,
            ),
            yaxis=dict(
                title="North (m)", gridcolor="rgba(148,163,184,0.1)",
                zeroline=False, showgrid=True, scaleanchor="x",
            ),
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("**Position Summary**")
        st.markdown(f"""
        <div class="sat-card">
        <span style="color:#94a3b8;">True</span><br>
        x = {result.true_position.x_m:.2f} m<br>
        y = {result.true_position.y_m:.2f} m
        </div>
        <div class="sat-card" style="border-color:rgba(248,113,113,0.4);">
        <span style="color:#f87171;">GPS Fix</span><br>
        x = {result.gps_position.x_m:.2f} m<br>
        y = {result.gps_position.y_m:.2f} m<br>
        Δ = <b style="color:#f87171;">{gps_err:.2f} m</b>
        </div>
        <div class="sat-card" style="border-color:rgba(52,211,153,0.4);">
        <span style="color:#34d399;">DGPS Fix</span><br>
        x = {result.dgps_position.x_m:.2f} m<br>
        y = {result.dgps_position.y_m:.2f} m<br>
        Δ = <b style="color:#34d399;">{dgps_err:.2f} m</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**DOP Values**")
        dop = result.dop_values
        for k, v in dop.items():
            color = "#34d399" if v < 2 else "#fbbf24" if v < 4 else "#f87171"
            st.markdown(
                f'<div class="step-box">'
                f'{k}: <span style="color:{color}; font-weight:700;">{v:.2f}</span></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SKY PLOT & DOP
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    col_sky, col_dop = st.columns([1, 1])

    with col_sky:
        st.markdown("### Sky Plot")
        st.caption("Satellite positions as seen from receiver. Centre = zenith, edge = horizon.")

        fig_sky = go.Figure()

        # Elevation rings
        for el_ring in [10, 30, 60, 90]:
            theta = np.linspace(0, 2*np.pi, 100)
            r = 1 - el_ring/90
            fig_sky.add_trace(go.Scatterpolar(
                r=[r]*100, theta=np.degrees(theta),
                mode="lines", showlegend=False,
                line=dict(color="rgba(148,163,184,0.2)", width=1),
            ))
            fig_sky.add_annotation(
                x=0.5 + r/2 * 0.52, y=0.5,
                text=f"{el_ring}°", showarrow=False,
                font=dict(size=9, color="#64748b"),
                xref="paper", yref="paper",
            )

        # Satellites
        for sat in result.satellites:
            r_plot = 1 - sat.elevation_deg / 90
            error_mag = abs(sat.total_error_m)
            color = f"rgba({min(255,int(error_mag*15))},{max(0,255-int(error_mag*15))},100,0.9)"

            fig_sky.add_trace(go.Scatterpolar(
                r=[r_plot], theta=[sat.azimuth_deg],
                mode="markers+text",
                text=[f"G{sat.prn:02d}"],
                textposition="top center",
                textfont=dict(size=9, color="#cbd5e1"),
                marker=dict(size=18, color=color,
                            line=dict(color="white", width=1)),
                name=f"PRN {sat.prn} ({sat.elevation_deg:.0f}°)",
                showlegend=False,
            ))

        fig_sky.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0, 1], showticklabels=False,
                    gridcolor="rgba(148,163,184,0.15)",
                ),
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
        st.caption("🔴 High error  🟢 Low error  (colour by total pseudorange error)")

    with col_dop:
        st.markdown("### DOP Gauges")
        st.caption("Dilution of Precision — measures geometric quality of satellite spread")

        dop_data = result.dop_values
        dop_info = {
            "GDOP": ("Geometric", "#a78bfa"),
            "PDOP": ("Position (3D)", "#38bdf8"),
            "HDOP": ("Horizontal", "#34d399"),
            "VDOP": ("Vertical", "#fbbf24"),
            "TDOP": ("Time", "#fb923c"),
        }

        fig_dop = go.Figure()
        for i, (key, (label, color)) in enumerate(dop_info.items()):
            val = min(dop_data[key], 10)
            fig_dop.add_trace(go.Bar(
                x=[val], y=[f"{key}\n{label}"],
                orientation="h",
                marker=dict(
                    color=color,
                    opacity=0.8,
                    line=dict(color=color, width=1),
                ),
                name=key,
                text=[f"{dop_data[key]:.2f}"],
                textposition="outside",
                textfont=dict(color="#f1f5f9"),
                showlegend=False,
            ))

        # DOP quality zones
        for x_val, label, color in [(1,"Ideal","#34d399"), (2,"Excellent","#86efac"),
                                     (5,"Good","#fbbf24"), (10,"Poor","#f87171")]:
            fig_dop.add_vline(x=x_val, line_dash="dot", line_color=color,
                               annotation_text=label,
                               annotation_font_color=color, annotation_font_size=10)

        fig_dop.update_layout(
            plot_bgcolor="#0d1526",
            paper_bgcolor="#0d1526",
            font=dict(color="#cbd5e1", family="Space Mono"),
            xaxis=dict(range=[0, 10], title="DOP Value",
                       gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.05)"),
            height=300,
            margin=dict(l=20, r=80, t=20, b=20),
        )
        st.plotly_chart(fig_dop, use_container_width=True)

        # DOP explanation
        st.markdown("""
        <div class="glass-card">
        <div class="concept-header">DOP Interpretation</div>
        <div style="font-family:'Space Mono',monospace; font-size:12px; color:#94a3b8; line-height:1.8;">
        &lt; 1 &nbsp; → Ideal<br>
        1–2 &nbsp; → Excellent<br>
        2–5 &nbsp; → Good (most receivers)<br>
        5–10 → Moderate<br>
        &gt; 10 → Poor — avoid if possible
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
        DOP = σ_position / σ_pseudorange<br><br>
        Low DOP = satellites well-spread in sky<br>
        High DOP = satellites clustered together
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ERROR BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Per-Satellite Error Breakdown")
    st.caption("GPS error budget decomposed by source for each satellite")

    error_types = ["Satellite Clock", "Ionospheric Delay", "Tropospheric Delay",
                   "Multipath", "Receiver Noise"]
    colors      = ["#a78bfa", "#38bdf8", "#34d399", "#fbbf24", "#fb923c"]

    # Build data
    sat_labels = [f"G{s.prn:02d}" for s in result.satellites]
    fig_err = go.Figure()

    for err_type, color in zip(error_types, colors):
        values = []
        for sat in result.satellites:
            eb = sat.error_breakdown()
            values.append(abs(eb[err_type]))

        fig_err.add_trace(go.Bar(
            name=err_type,
            x=sat_labels,
            y=values,
            marker_color=color,
            opacity=0.85,
        ))

    fig_err.update_layout(
        barmode="stack",
        plot_bgcolor="#0d1526",
        paper_bgcolor="#0d1526",
        font=dict(color="#cbd5e1", family="Space Mono"),
        legend=dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="rgba(99,179,237,0.3)",
                    borderwidth=1),
        xaxis=dict(title="Satellite PRN", gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(title="Error (m)", gridcolor="rgba(148,163,184,0.1)"),
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # Scatter: elevation vs total error
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Elevation vs Error** (atmospheric mapping effect)")
        fig_el = go.Figure()
        for sat in result.satellites:
            total = abs(sat.total_error_m)
            atm   = abs(sat.ionospheric_error_m + sat.tropospheric_error_m)
            fig_el.add_trace(go.Scatter(
                x=[sat.elevation_deg], y=[total],
                mode="markers+text",
                text=[f"G{sat.prn}"],
                textposition="top center",
                textfont=dict(size=9, color="#94a3b8"),
                marker=dict(size=12, color="#38bdf8", opacity=0.8),
                name=f"G{sat.prn}", showlegend=False,
            ))

        el_range = np.linspace(10, 85, 50)
        mapping = 1 / np.sin(np.radians(el_range))
        nominal = (7*iono_scale + 2.3*tropo_scale) * mapping
        fig_el.add_trace(go.Scatter(
            x=el_range, y=nominal,
            mode="lines", name="Atm. mapping curve",
            line=dict(color="#fbbf24", dash="dash", width=1.5),
        ))

        fig_el.update_layout(
            plot_bgcolor="#0d1526", paper_bgcolor="#0d1526",
            font=dict(color="#cbd5e1", family="Space Mono"),
            xaxis=dict(title="Elevation (°)", gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title="Total Error (m)", gridcolor="rgba(148,163,184,0.1)"),
            height=300, margin=dict(l=10,r=10,t=10,b=10),
        )
        st.plotly_chart(fig_el, use_container_width=True)

    with col_b:
        st.markdown("**Error Source Pie** (average across satellites)")
        avg_errors = {k: 0.0 for k in error_types}
        for sat in result.satellites:
            for k, v in sat.error_breakdown().items():
                avg_errors[k] += abs(v)
        for k in avg_errors:
            avg_errors[k] /= len(result.satellites)

        fig_pie = go.Figure(go.Pie(
            labels=list(avg_errors.keys()),
            values=list(avg_errors.values()),
            marker=dict(colors=colors, line=dict(color="#0d1526", width=2)),
            hole=0.4,
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0d1526",
            font=dict(color="#cbd5e1", family="Space Mono"),
            legend=dict(bgcolor="rgba(13,21,38,0.8)", font=dict(size=10)),
            height=300, margin=dict(l=10,r=10,t=10,b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: DGPS CORRECTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### DGPS Correction Vectors")
    st.caption("How the reference station isolates and broadcasts corrections")

    col_corr, col_table = st.columns([2, 1])

    with col_corr:
        prns         = [f"G{s.prn:02d}" for s in result.satellites]
        corrections  = result.dgps_correction_vector

        fig_corr = make_subplots(rows=2, cols=1,
                                  subplot_titles=["Pseudorange Error Before/After Correction",
                                                  "Correction Applied per Satellite"],
                                  vertical_spacing=0.18)

        before = [abs(result.satellites[i].total_error_m) for i in range(len(result.satellites))]
        after  = [abs(result.satellites[i].multipath_error_m +
                      result.satellites[i].receiver_noise_m)
                  for i in range(len(result.satellites))]

        fig_corr.add_trace(go.Bar(x=prns, y=before, name="GPS (no correction)",
                                   marker_color="#f87171", opacity=0.8), row=1, col=1)
        fig_corr.add_trace(go.Bar(x=prns, y=after, name="After DGPS correction",
                                   marker_color="#34d399", opacity=0.8), row=1, col=1)

        corr_vals = [abs(corrections[s.prn]["correction_m"]) for s in result.satellites]
        fig_corr.add_trace(go.Bar(x=prns, y=corr_vals, name="Correction magnitude",
                                   marker_color="#38bdf8", opacity=0.8), row=2, col=1)

        fig_corr.update_layout(
            plot_bgcolor="#0d1526", paper_bgcolor="#0d1526",
            font=dict(color="#cbd5e1", family="Space Mono"),
            barmode="group",
            height=480,
            legend=dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="rgba(99,179,237,0.3)",
                        borderwidth=1),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        for i in range(1, 3):
            fig_corr.update_xaxes(gridcolor="rgba(148,163,184,0.1)", row=i, col=1)
            fig_corr.update_yaxes(title_text="Metres", gridcolor="rgba(148,163,184,0.1)",
                                   row=i, col=1)

        st.plotly_chart(fig_corr, use_container_width=True)

    with col_table:
        st.markdown("**Correction Detail**")
        rows = []
        for sat in result.satellites:
            c = corrections[sat.prn]
            rows.append({
                "PRN": f"G{sat.prn:02d}",
                "Total Err (m)": f"{sat.total_error_m:.2f}",
                "Correction (m)": f"{c['correction_m']:.2f}",
                "Residual (m)": f"{sat.multipath_error_m + sat.receiver_noise_m:.2f}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)

        st.markdown("""
        <div class="formula-box">
        DGPS Correction:<br><br>
        Δρ = ρ_measured − ρ_true<br><br>
        Broadcast to rover.<br>
        Rover applies: ρ_corrected = ρ_rover − Δρ
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="font-size:12px; font-family:'Space Mono',monospace; color:#94a3b8;">
        ✅ Removes: Clock, Iono, Tropo<br>
        ❌ Can't remove: Multipath, Noise<br>
        ⚠️ Accuracy degrades with distance
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(f"### Monte Carlo Analysis — {len(mc_df)} Simulation Runs")

    col_hist, col_box = st.columns(2)

    with col_hist:
        st.markdown("**Error Distribution**")
        fig_hist = go.Figure()
        for col, color, name in [
            ("GPS Error (m)", "#f87171", "GPS"),
            ("DGPS Error (m)", "#34d399", "DGPS"),
        ]:
            fig_hist.add_trace(go.Histogram(
                x=mc_df[col], name=name, opacity=0.75,
                marker_color=color, nbinsx=30,
            ))
        fig_hist.update_layout(
            barmode="overlay",
            plot_bgcolor="#0d1526", paper_bgcolor="#0d1526",
            font=dict(color="#cbd5e1", family="Space Mono"),
            xaxis=dict(title="Position Error (m)", gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(title="Count", gridcolor="rgba(148,163,184,0.1)"),
            legend=dict(bgcolor="rgba(13,21,38,0.8)"),
            height=340, margin=dict(l=10,r=10,t=10,b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        st.markdown("**Box Plot Comparison**")
        fig_box = go.Figure()
        for col, color, name in [
            ("GPS Error (m)", "#f87171", "GPS"),
            ("DGPS Error (m)", "#34d399", "DGPS"),
        ]:
            fig_box.add_trace(go.Box(
                y=mc_df[col], name=name,
                marker_color=color,
                line_color=color,
                fillcolor=color.replace(")", ",0.2)").replace("rgb", "rgba"),
            ))
        fig_box.update_layout(
            plot_bgcolor="#0d1526", paper_bgcolor="#0d1526",
            font=dict(color="#cbd5e1", family="Space Mono"),
            yaxis=dict(title="Error (m)", gridcolor="rgba(148,163,184,0.1)"),
            height=340, margin=dict(l=10,r=10,t=10,b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Time series
    st.markdown("**Error Over Simulation Runs**")
    fig_ts = go.Figure()
    for col, color, name in [
        ("GPS Error (m)", "#f87171", "GPS"),
        ("DGPS Error (m)", "#34d399", "DGPS"),
    ]:
        fig_ts.add_trace(go.Scatter(
            x=mc_df["run"], y=mc_df[col].rolling(10).mean(),
            name=f"{name} (10-run avg)", line=dict(color=color, width=2),
        ))
        fig_ts.add_trace(go.Scatter(
            x=mc_df["run"], y=mc_df[col],
            name=f"{name} raw", line=dict(color=color, width=0.5),
            opacity=0.3, showlegend=False,
        ))
    fig_ts.update_layout(
        plot_bgcolor="#0d1526", paper_bgcolor="#0d1526",
        font=dict(color="#cbd5e1", family="Space Mono"),
        xaxis=dict(title="Run #", gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(title="Error (m)", gridcolor="rgba(148,163,184,0.1)"),
        legend=dict(bgcolor="rgba(13,21,38,0.8)"),
        height=280, margin=dict(l=10,r=10,t=10,b=10),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Summary stats
    st.markdown("**Summary Statistics**")
    stats = mc_df[["GPS Error (m)", "DGPS Error (m)"]].describe().round(3)
    st.dataframe(stats, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    col_gps, col_dgps = st.columns(2)

    with col_gps:
        st.markdown("## 📡 How GPS Works")
        st.markdown("""
        <div class="glass-card">
        <div class="concept-header">1. Trilateration</div>
        GPS works by measuring the time it takes for signals from multiple satellites
        to reach the receiver. Since signals travel at the speed of light, distance
        = time × c. With 4+ satellites, the receiver solves for x, y, z, and clock bias.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
        Pseudorange equation:<br>
        ρᵢ = |Xᵢ − X| + c·δt + εᵢ<br><br>
        Where:<br>
        ρᵢ = measured pseudorange to sat i<br>
        Xᵢ = satellite position (known)<br>
        X  = receiver position (unknown)<br>
        δt = receiver clock offset<br>
        εᵢ = all errors combined
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <div class="concept-header">2. Error Sources</div>
        <table style="font-family:'Space Mono',monospace; font-size:12px; width:100%; color:#94a3b8;">
        <tr><td style="color:#a78bfa;">Clock errors</td><td>~2–5 m</td><td>Satellite timing drift</td></tr>
        <tr><td style="color:#38bdf8;">Ionosphere</td><td>~5–15 m</td><td>Free electrons delay signal</td></tr>
        <tr><td style="color:#34d399;">Troposphere</td><td>~0.5–3 m</td><td>Water vapour, pressure</td></tr>
        <tr><td style="color:#fbbf24;">Multipath</td><td>~0–5 m</td><td>Signal reflections</td></tr>
        <tr><td style="color:#fb923c;">Receiver noise</td><td>~0.3 m</td><td>Thermal noise in chip</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <div class="concept-header">3. Least-Squares Solution</div>
        The receiver linearises the equations and iteratively solves:
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
        Δx = (AᵀA)⁻¹ Aᵀ Δρ<br><br>
        A = design matrix (direction cosines)<br>
        Δρ = pseudorange residuals<br>
        Δx = position correction vector<br><br>
        Repeat until |Δx| < threshold
        </div>
        """, unsafe_allow_html=True)

    with col_dgps:
        st.markdown("## 🛰️ How DGPS Works")
        st.markdown("""
        <div class="glass-card">
        <div class="concept-header">Key Insight</div>
        A reference station at a <b>known exact position</b> can compute the error in
        every GPS pseudorange. Since atmospheric errors are spatially correlated
        (nearby receivers see the same ionosphere), broadcasting these corrections
        dramatically improves rover accuracy.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
        At reference station:<br>
        Δρᵢ = ρᵢ_measured − ρᵢ_true<br><br>
        Δρᵢ ≈ clock_err + iono + tropo<br><br>
        At rover (user receiver):<br>
        ρᵢ_corrected = ρᵢ_rover − Δρᵢ<br><br>
        Remaining errors: multipath + noise
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <div class="concept-header">What DGPS Removes</div>
        <table style="font-family:'Space Mono',monospace; font-size:12px; width:100%; color:#94a3b8;">
        <tr><td style="color:#34d399;">✅ Satellite clocks</td><td>Completely removed</td></tr>
        <tr><td style="color:#34d399;">✅ Ionospheric delay</td><td>~90% removed (if &lt;50 km)</td></tr>
        <tr><td style="color:#34d399;">✅ Tropospheric delay</td><td>~80% removed</td></tr>
        <tr><td style="color:#f87171;">❌ Multipath</td><td>Site-specific, not correlated</td></tr>
        <tr><td style="color:#f87171;">❌ Receiver noise</td><td>Independent at each receiver</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <div class="concept-header">DGPS vs RTK vs PPP</div>
        <table style="font-family:'Space Mono',monospace; font-size:12px; width:100%; color:#94a3b8;">
        <tr><th style="color:#f1f5f9;">System</th><th style="color:#f1f5f9;">Accuracy</th><th style="color:#f1f5f9;">Range</th></tr>
        <tr><td>GPS alone</td><td>3–10 m</td><td>Global</td></tr>
        <tr><td style="color:#38bdf8;">DGPS</td><td>0.5–3 m</td><td>&lt;300 km</td></tr>
        <tr><td style="color:#a78bfa;">RTK</td><td>1–3 cm</td><td>&lt;30 km</td></tr>
        <tr><td style="color:#34d399;">PPP</td><td>2–10 cm</td><td>Global</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <div class="concept-header">Real-World Applications</div>
        <div style="font-family:'Space Mono',monospace; font-size:12px; color:#94a3b8; line-height:2;">
        🚢 Maritime navigation (USCG beacons)<br>
        ✈️ Aircraft approach guidance (GBAS)<br>
        🚜 Precision agriculture (&lt;30 cm rows)<br>
        🗺️ Survey-grade mapping<br>
        🚗 Autonomous vehicle lane-keeping<br>
        ⛽ Offshore platform positioning
        </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: STEP-BY-STEP MATH
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### Step-by-Step Computation Log")
    st.caption("Every step of the least-squares position fix — glass box mode")

    for step in result.step_log:
        if step.startswith("##"):
            st.markdown(step)
        elif step == "---":
            st.markdown("---")
        else:
            st.markdown(f'<div class="step-box">{step.replace("**","<b>").replace("**","</b>")}</div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Satellite Data Table")
    sat_rows = []
    for sat in result.satellites:
        sat_rows.append({
            "PRN":       f"G{sat.prn:02d}",
            "El (°)":    f"{sat.elevation_deg:.1f}",
            "Az (°)":    f"{sat.azimuth_deg:.1f}",
            "True Range (m)": f"{sat.true_range_m:.1f}",
            "Clock (m)": f"{sat.clock_error_m:.3f}",
            "Iono (m)":  f"{sat.ionospheric_error_m:.3f}",
            "Tropo (m)": f"{sat.tropospheric_error_m:.3f}",
            "Multipath (m)": f"{sat.multipath_error_m:.3f}",
            "Noise (m)": f"{sat.receiver_noise_m:.3f}",
            "Total Err (m)": f"{sat.total_error_m:.3f}",
            "Pseudorange (m)": f"{sat.pseudorange_m:.3f}",
        })
    st.dataframe(pd.DataFrame(sat_rows), use_container_width=True)

    st.markdown("### Atmospheric Mapping Function")
    st.markdown("""
    <div class="formula-box">
    Mapping function: M(El) = 1 / sin(El)<br><br>
    This scales the zenith delay to the actual slant path through the atmosphere.<br>
    At El=90° (overhead): M = 1.0  (minimum path length)<br>
    At El=10° (horizon):  M = 5.76 (5.76× longer path through atmosphere)<br><br>
    Ionospheric zenith delay ≈ 7 m × scale_factor × M(El)<br>
    Tropospheric zenith delay ≈ 2.3 m × scale_factor × M(El)
    </div>
    """, unsafe_allow_html=True)

    el_vals = np.arange(10, 91, 1)
    map_vals = 1 / np.sin(np.radians(el_vals))
    fig_map = go.Figure()
    fig_map.add_trace(go.Scatter(
        x=el_vals, y=map_vals,
        mode="lines", name="M(El)",
        line=dict(color="#38bdf8", width=2),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.1)",
    ))
    fig_map.update_layout(
        plot_bgcolor="#0d1526", paper_bgcolor="#0d1526",
        font=dict(color="#cbd5e1", family="Space Mono"),
        xaxis=dict(title="Elevation (°)", gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(title="Mapping Factor", gridcolor="rgba(148,163,184,0.1)"),
        height=280, margin=dict(l=20,r=20,t=10,b=20),
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:32px 0 16px; 
     font-family:'Space Mono',monospace; font-size:11px; 
     letter-spacing:2px; color:#334155;">
    GPS & DGPS TEACHING SIMULATOR · GLASS-BOX EDITION<br>
    All physics, all math, all visible — built for education
</div>
""", unsafe_allow_html=True)
