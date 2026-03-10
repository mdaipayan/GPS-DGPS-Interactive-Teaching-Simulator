"""
GPS and DGPS Core Simulation Engine
Glass-Box Teaching Module — all math is visible and explained
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import pandas as pd


# ─── Constants ──────────────────────────────────────────────────────────────
SPEED_OF_LIGHT = 299_792_458.0        # m/s
EARTH_RADIUS   = 6_371_000.0          # m
GPS_FREQ_L1    = 1575.42e6            # Hz
GPS_WAVELENGTH = SPEED_OF_LIGHT / GPS_FREQ_L1


# ─── Data Classes ────────────────────────────────────────────────────────────
@dataclass
class Satellite:
    """Represents a GPS satellite in orbit"""
    prn: int                    # Pseudo-random noise code ID
    elevation_deg: float        # Elevation above horizon (degrees)
    azimuth_deg: float          # Azimuth from North (degrees)
    true_range_m: float         # True geometric range to receiver (m)

    # Error components (glass-box: each source is separated)
    clock_error_m: float = 0.0          # Satellite clock bias
    ionospheric_error_m: float = 0.0    # Ionospheric delay
    tropospheric_error_m: float = 0.0   # Tropospheric delay
    multipath_error_m: float = 0.0      # Multipath reflection error
    receiver_noise_m: float = 0.0       # Receiver thermal noise

    @property
    def total_error_m(self) -> float:
        return (self.clock_error_m + self.ionospheric_error_m +
                self.tropospheric_error_m + self.multipath_error_m +
                self.receiver_noise_m)

    @property
    def pseudorange_m(self) -> float:
        """What the GPS receiver actually measures"""
        return self.true_range_m + self.total_error_m

    @property
    def pseudorange_corrected_m(self) -> float:
        """After DGPS correction (removes most errors except noise & multipath)"""
        return self.true_range_m + self.multipath_error_m + self.receiver_noise_m

    def error_breakdown(self) -> dict:
        return {
            "Satellite Clock": self.clock_error_m,
            "Ionospheric Delay": self.ionospheric_error_m,
            "Tropospheric Delay": self.tropospheric_error_m,
            "Multipath": self.multipath_error_m,
            "Receiver Noise": self.receiver_noise_m,
        }


@dataclass
class ReceiverPosition:
    """2D position (lat/lon in metres offset from origin)"""
    x_m: float = 0.0
    y_m: float = 0.0

    def distance_to(self, other: "ReceiverPosition") -> float:
        return np.sqrt((self.x_m - other.x_m)**2 + (self.y_m - other.y_m)**2)

    def to_latlon(self, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
        lat = origin_lat + (self.y_m / 111_111)
        lon = origin_lon + (self.x_m / (111_111 * np.cos(np.radians(origin_lat))))
        return lat, lon


@dataclass
class GPSSimulationResult:
    """Full result of one GPS fix — every number is traceable"""
    true_position: ReceiverPosition
    gps_position: ReceiverPosition
    dgps_position: ReceiverPosition
    satellites: List[Satellite]
    dop_values: dict
    dgps_correction_vector: dict      # correction applied per satellite
    step_log: List[str]               # human-readable computation steps


# ─── Simulation Engine ────────────────────────────────────────────────────────
class GPSSimulator:
    """
    Glass-Box GPS/DGPS simulator.
    All intermediate calculations are stored for teaching purposes.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ── Satellite Constellation ─────────────────────────────────────────────
    def generate_constellation(
        self,
        n_satellites: int = 8,
        min_elevation: float = 10.0,
    ) -> List[Satellite]:
        """Generate a realistic constellation visible from mid-latitudes"""
        satellites = []
        elevations = self.rng.uniform(min_elevation, 85, n_satellites)
        azimuths   = np.linspace(0, 360, n_satellites, endpoint=False)
        azimuths  += self.rng.uniform(-15, 15, n_satellites)

        for i, (el, az) in enumerate(zip(elevations, azimuths)):
            # True range based on elevation (higher elevation → shorter path through atm)
            slant_range = EARTH_RADIUS * (
                np.sqrt((20_200_000 / EARTH_RADIUS)**2 - np.cos(np.radians(el))**2)
                - np.sin(np.radians(el))
            )
            sat = Satellite(
                prn=i + 1,
                elevation_deg=float(el),
                azimuth_deg=float(az % 360),
                true_range_m=float(slant_range),
            )
            satellites.append(sat)
        return satellites

    # ── Error Models ────────────────────────────────────────────────────────
    def apply_errors(
        self,
        satellites: List[Satellite],
        ionospheric_scale: float = 1.0,
        tropospheric_scale: float = 1.0,
        multipath_scale: float = 1.0,
        clock_scale: float = 1.0,
    ) -> List[Satellite]:
        """
        Apply realistic error models to each satellite.
        Errors scale with 1/sin(elevation) — lower satellites have more atmosphere.
        """
        for sat in satellites:
            el_rad = np.radians(sat.elevation_deg)
            mapping = 1.0 / np.sin(el_rad)   # atmospheric mapping function

            # Clock error: common bias + per-satellite variation
            sat.clock_error_m = clock_scale * (
                self.rng.normal(2.0, 0.5) + self.rng.uniform(-1, 1)
            )

            # Ionospheric: L1 single-frequency, Klobuchar model approximated
            iono_zenith = 7.0 * ionospheric_scale   # metres at zenith
            sat.ionospheric_error_m = iono_zenith * mapping * self.rng.uniform(0.8, 1.2)

            # Tropospheric: Hopfield model approximated
            tropo_zenith = 2.3 * tropospheric_scale
            sat.tropospheric_error_m = tropo_zenith * mapping * self.rng.uniform(0.9, 1.1)

            # Multipath: depends on environment, not correlated with elevation simply
            sat.multipath_error_m = multipath_scale * self.rng.normal(0, 1.5)

            # Receiver noise: white noise
            sat.receiver_noise_m = self.rng.normal(0, 0.3)

        return satellites

    # ── DOP Calculation ─────────────────────────────────────────────────────
    def compute_dop(self, satellites: List[Satellite]) -> dict:
        """
        Compute Dilution of Precision from satellite geometry.
        H = (A^T A)^{-1}  where A is the direction cosine matrix.
        """
        rows = []
        for sat in satellites:
            el = np.radians(sat.elevation_deg)
            az = np.radians(sat.azimuth_deg)
            # Unit vector from receiver to satellite in ENU frame
            e = np.cos(el) * np.sin(az)
            n = np.cos(el) * np.cos(az)
            u = np.sin(el)
            rows.append([e, n, u, 1.0])   # last column: clock

        A = np.array(rows)
        try:
            H = np.linalg.inv(A.T @ A)
            HDOP = np.sqrt(H[0, 0] + H[1, 1])
            VDOP = np.sqrt(H[2, 2])
            PDOP = np.sqrt(H[0, 0] + H[1, 1] + H[2, 2])
            TDOP = np.sqrt(H[3, 3])
            GDOP = np.sqrt(np.trace(H))
        except np.linalg.LinAlgError:
            HDOP = VDOP = PDOP = TDOP = GDOP = float("inf")

        return {"HDOP": HDOP, "VDOP": VDOP, "PDOP": PDOP, "TDOP": TDOP, "GDOP": GDOP}

    # ── Least-Squares Position Fix ──────────────────────────────────────────
    def compute_position_fix(
        self,
        satellites: List[Satellite],
        use_corrections: bool = False,
        true_x: float = 0.0,
        true_y: float = 0.0,
    ) -> Tuple[ReceiverPosition, List[str]]:
        """
        Iterative least-squares receiver position estimate.
        Returns position and a step-by-step log for teaching.
        """
        steps = []
        # Initial guess: true position + random offset to simulate cold start
        x, y = true_x + self.rng.normal(0, 50), true_y + self.rng.normal(0, 50)
        z = 0.0
        b = 0.0  # receiver clock bias

        steps.append(f"**Initial guess:** x={x:.1f} m, y={y:.1f} m, clock_bias={b:.3f} m")
        steps.append(f"**Number of satellites used:** {len(satellites)}")

        for iteration in range(8):
            residuals = []
            H_rows = []

            for sat in satellites:
                pr = sat.pseudorange_corrected_m if use_corrections else sat.pseudorange_m

                # Predicted range from current estimate
                # Using simplified 2D: project sat position
                el = np.radians(sat.elevation_deg)
                az = np.radians(sat.azimuth_deg)
                sx = sat.true_range_m * np.cos(el) * np.sin(az) + true_x
                sy = sat.true_range_m * np.cos(el) * np.cos(az) + true_y
                sz = sat.true_range_m * np.sin(el)

                predicted_r = np.sqrt((x-sx)**2 + (y-sy)**2 + (z-sz)**2) + b
                delta_rho   = pr - predicted_r

                # Direction cosines (design matrix row)
                r = max(predicted_r - b, 1.0)
                h = [(x-sx)/r, (y-sy)/r, (z-sz)/r, 1.0]

                residuals.append(delta_rho)
                H_rows.append(h)

            H = np.array(H_rows)
            rho = np.array(residuals)

            try:
                delta = np.linalg.lstsq(H, rho, rcond=None)[0]
            except Exception:
                break

            x += delta[0]
            y += delta[1]
            z += delta[2]
            b += delta[3]

            correction_norm = np.sqrt(delta[0]**2 + delta[1]**2)
            if correction_norm < 0.001:
                steps.append(f"**Converged at iteration {iteration+1}** (Δ={correction_norm:.4f} m)")
                break

        steps.append(f"**Final fix:** x={x:.2f} m, y={y:.2f} m, clock_bias={b:.4f} m")
        return ReceiverPosition(x_m=x, y_m=y), steps

    # ── DGPS Correction Engine ───────────────────────────────────────────────
    def compute_dgps_corrections(
        self,
        satellites: List[Satellite],
        ref_x: float,
        ref_y: float,
    ) -> dict:
        """
        Reference station knows its exact position.
        It computes the error in each pseudorange and broadcasts corrections.

        DGPS correction = True_range - Pseudorange  (at reference station)
        """
        corrections = {}
        for sat in satellites:
            el = np.radians(sat.elevation_deg)
            az = np.radians(sat.azimuth_deg)
            sx = sat.true_range_m * np.cos(el) * np.sin(az) + ref_x
            sy = sat.true_range_m * np.cos(el) * np.cos(az) + ref_y
            sz = sat.true_range_m * np.sin(el)

            true_range_ref = np.sqrt(sx**2 + sy**2 + sz**2)
            # Correction = what should be subtracted from pseudorange
            correction = sat.pseudorange_m - true_range_ref
            corrections[sat.prn] = {
                "correction_m": correction,
                "removed_errors": {
                    "clock": sat.clock_error_m,
                    "iono": sat.ionospheric_error_m,
                    "tropo": sat.tropospheric_error_m,
                }
            }
        return corrections

    # ── Full Simulation Run ──────────────────────────────────────────────────
    def run(
        self,
        true_x: float = 0.0,
        true_y: float = 0.0,
        n_satellites: int = 8,
        ionospheric_scale: float = 1.0,
        tropospheric_scale: float = 1.0,
        multipath_scale: float = 1.0,
        clock_scale: float = 1.0,
        ref_station_offset_km: float = 10.0,
    ) -> GPSSimulationResult:
        """Run full GPS + DGPS simulation, returning all intermediate results."""

        # 1. Generate constellation
        sats = self.generate_constellation(n_satellites)

        # 2. Apply errors
        sats = self.apply_errors(
            sats, ionospheric_scale, tropospheric_scale, multipath_scale, clock_scale
        )

        # 3. GPS fix (no corrections)
        gps_pos, gps_steps = self.compute_position_fix(sats, False, true_x, true_y)

        # 4. DGPS reference station corrections
        ref_x = true_x + ref_station_offset_km * 1000
        ref_y = true_y
        corrections = self.compute_dgps_corrections(sats, ref_x, ref_y)

        # Apply corrections to satellites (modifies pseudorange used in fix)
        corrected_sats = []
        for sat in sats:
            corr = corrections[sat.prn]["correction_m"]
            new_sat = Satellite(
                prn=sat.prn,
                elevation_deg=sat.elevation_deg,
                azimuth_deg=sat.azimuth_deg,
                true_range_m=sat.true_range_m,
                clock_error_m=0.0,
                ionospheric_error_m=0.0,
                tropospheric_error_m=0.0,
                multipath_error_m=sat.multipath_error_m,
                receiver_noise_m=sat.receiver_noise_m,
            )
            corrected_sats.append(new_sat)

        # 5. DGPS fix (with corrections)
        dgps_pos, dgps_steps = self.compute_position_fix(corrected_sats, True, true_x, true_y)

        # 6. DOP
        dop = self.compute_dop(sats)

        step_log = (
            ["## 📡 GPS Fix Steps"] + gps_steps +
            ["---", "## 🛰️ DGPS Fix Steps"] + dgps_steps
        )

        return GPSSimulationResult(
            true_position=ReceiverPosition(true_x, true_y),
            gps_position=gps_pos,
            dgps_position=dgps_pos,
            satellites=sats,
            dop_values=dop,
            dgps_correction_vector=corrections,
            step_log=step_log,
        )


# ── Monte Carlo for Statistical Teaching ─────────────────────────────────────
def monte_carlo_simulation(
    n_runs: int = 200,
    n_satellites: int = 8,
    ionospheric_scale: float = 1.0,
    tropospheric_scale: float = 1.0,
    multipath_scale: float = 1.0,
) -> pd.DataFrame:
    """Run many simulations to build statistical picture of GPS vs DGPS accuracy"""
    records = []
    sim = GPSSimulator()

    for i in range(n_runs):
        sim.rng = np.random.default_rng(i)
        result = sim.run(
            n_satellites=n_satellites,
            ionospheric_scale=ionospheric_scale,
            tropospheric_scale=tropospheric_scale,
            multipath_scale=multipath_scale,
        )
        gps_err  = result.true_position.distance_to(result.gps_position)
        dgps_err = result.true_position.distance_to(result.dgps_position)
        records.append({
            "run": i,
            "GPS Error (m)": gps_err,
            "DGPS Error (m)": dgps_err,
            "HDOP": result.dop_values["HDOP"],
            "PDOP": result.dop_values["PDOP"],
        })

    return pd.DataFrame(records)
