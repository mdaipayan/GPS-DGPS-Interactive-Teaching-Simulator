# 🛰️ GPS & DGPS Teaching Simulator — Glass-Box Edition

An interactive, transparent simulation of GPS and Differential GPS (DGPS) positioning for classroom teaching. Every calculation is exposed — no black boxes.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 What It Simulates

### GPS Standard Positioning
- Realistic satellite constellation generation (configurable count, min elevation)
- Full error budget modelling with **separated sources**:
  - Satellite clock bias (~2–5 m)
  - Ionospheric delay (Klobuchar-approximated, ~5–15 m at low elevation)
  - Tropospheric delay (Hopfield-approximated, ~0.5–3 m)
  - Multipath reflections (~0–5 m, random)
  - Receiver thermal noise (~0.3 m RMS)
- Iterative **Weighted Least-Squares** position fix
- Full **DOP** computation (HDOP, VDOP, PDOP, TDOP, GDOP)

### DGPS (Differential GPS)
- Reference station correction computation
- Pseudorange correction broadcast (RTCM-style concept)
- Rover correction application
- Residual error analysis (multipath + noise remain)
- Distance-dependent decorrelation of corrections

---

## 📚 Teaching Tabs

| Tab | Content |
|-----|---------|
| 🗺️ Position Map | Visual comparison of GPS vs DGPS fix vs truth |
| 📡 Sky Plot & DOP | Polar sky plot + DOP bar gauges |
| ⚠️ Error Breakdown | Stacked bar chart per satellite per error source |
| 🔧 DGPS Corrections | Before/after correction per PRN, correction table |
| 📊 Statistics | Monte Carlo histogram, box plot, time series |
| 📚 How It Works | Conceptual explanation with formulas |
| 🔬 Step-by-Step Math | Full least-squares iteration log + data table |

---

## 🚀 Quick Start

### Option 1 — Local

```bash
git clone https://github.com/YOUR_USERNAME/gps-dgps-simulator.git
cd gps-dgps-simulator
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Option 2 — Streamlit Community Cloud

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your fork
4. Set **Main file path** to `app.py`
5. Click **Deploy** — it's free!

---

## 🔬 The Physics

### Pseudorange Equation

```
ρᵢ = |Xᵢ − X| + c·δt + δion + δtrop + δmulti + εᵢ
```

Where:
- `ρᵢ` = pseudorange measurement to satellite `i`
- `Xᵢ` = satellite ECEF position (from ephemeris)
- `X`  = receiver position (unknown)
- `c·δt` = receiver clock offset × speed of light
- `δion` = ionospheric delay (1/sin(el) mapping)
- `δtrop` = tropospheric delay (1/sin(el) mapping)
- `δmulti` = multipath error
- `εᵢ` = receiver noise

### Least-Squares Fix

```
Linearise: Δρ = A · Δx
Solution:  Δx = (AᵀA)⁻¹ Aᵀ Δρ
Iterate until |Δx| < 1 mm
```

Where `A` is the direction cosine matrix (unit vectors from receiver to each satellite).

### DOP Computation

```
H = (AᵀA)⁻¹
HDOP = √(H₁₁ + H₂₂)
VDOP = √(H₃₃)
PDOP = √(H₁₁ + H₂₂ + H₃₃)
GDOP = √(trace(H))
```

### Atmospheric Mapping Function

```
M(El) = 1 / sin(El)

Iono delay = 7 m × scale × M(El)
Tropo delay = 2.3 m × scale × M(El)
```

### DGPS Correction

At reference station (known position `X_ref`):
```
Δρᵢ_corr = ρᵢ_measured − |Xᵢ − X_ref|
```

At rover:
```
ρᵢ_fixed = ρᵢ_rover − Δρᵢ_corr
```

---

## 🎛️ Controls Reference

| Control | Effect |
|---------|--------|
| Number of Satellites | Constellation size (4 min for 3D fix) |
| Ionospheric Scale | Multiply default ~7m zenith iono delay |
| Tropospheric Scale | Multiply default ~2.3m zenith tropo delay |
| Multipath Scale | Scale random multipath noise |
| Clock Error Scale | Scale satellite clock bias |
| Ref Station Distance | Distance of DGPS reference station (km) |
| Random Seed | Reproducible satellite geometry |
| Monte Carlo Runs | Number of runs for statistical analysis |

---

## 📁 File Structure

```
gps-dgps-simulator/
├── app.py              # Main Streamlit application
├── gps_core.py         # Simulation engine (all math here)
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Dark theme configuration
└── README.md
```

---

## 🏫 Classroom Use

### Suggested Teaching Sequence

1. **Introduction** (Tab 6 — "How It Works")
   - Explain trilateration concept
   - Show the pseudorange equation
   - Discuss each error source

2. **Error Exploration** (Tab 3 — "Error Breakdown")
   - Increase ionospheric scale → watch errors grow at low-elevation sats
   - Show the 1/sin(El) mapping function (Tab 7)
   - Discuss why low-elevation satellites have larger errors

3. **Geometry Effects** (Tab 2 — "Sky Plot & DOP")
   - Reduce satellite count to 4 → watch DOP increase
   - Explain PDOP < 3 is good practice
   - Show HDOP vs VDOP relationship

4. **DGPS Mechanics** (Tab 4 — "DGPS Corrections")
   - Show what corrections remove vs what remains
   - Increase reference station distance → watch residuals grow

5. **Statistical Understanding** (Tab 5 — "Statistics")
   - Run Monte Carlo → discuss CEP (Circular Error Probable)
   - Compare GPS vs DGPS distributions
   - Introduce concept of accuracy vs precision

### Discussion Questions

- Why can't DGPS correct multipath errors?
- What happens to DGPS accuracy as reference station distance increases?
- Why is VDOP typically worse than HDOP?
- How would RTK improve upon DGPS?
- What is the minimum number of satellites needed for a 3D fix? Why?

---

## 📖 References

- Kaplan, E.D. & Hegarty, C.J. (2006). *Understanding GPS: Principles and Applications*
- Hofmann-Wellenhof, B. et al. (2008). *GNSS — Global Navigation Satellite Systems*
- IS-GPS-200 Interface Specification (ICD-GPS-200)
- RTCM Standard 10402 (DGPS correction format)

---

## 📄 License

MIT License — free to use, modify, and distribute for educational purposes.
