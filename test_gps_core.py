"""Unit tests for GPS/DGPS simulation core"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')
from gps_core import GPSSimulator, ReceiverPosition, monte_carlo_simulation

def test_constellation_generation():
    sim = GPSSimulator(seed=42)
    sats = sim.generate_constellation(8)
    assert len(sats) == 8
    for sat in sats:
        assert sat.elevation_deg >= 10
        assert 0 <= sat.azimuth_deg < 360
        assert sat.true_range_m > 0

def test_error_application():
    sim = GPSSimulator(seed=42)
    sats = sim.generate_constellation(8)
    sats = sim.apply_errors(sats)
    for sat in sats:
        assert sat.total_error_m != 0
        # Low elevation sats should generally have larger atmospheric errors
        assert abs(sat.ionospheric_error_m) >= 0

def test_dop_positive():
    sim = GPSSimulator(seed=42)
    sats = sim.generate_constellation(8)
    dop = sim.compute_dop(sats)
    for key in ["HDOP", "VDOP", "PDOP", "GDOP"]:
        assert dop[key] > 0

def test_position_fix_convergence():
    sim = GPSSimulator(seed=42)
    sats = sim.generate_constellation(8)
    sats = sim.apply_errors(sats)
    pos, steps = sim.compute_position_fix(sats, False, 0, 0)
    # Should produce a finite position
    assert np.isfinite(pos.x_m)
    assert np.isfinite(pos.y_m)

def test_dgps_improves_accuracy():
    """DGPS should on average produce smaller errors than GPS alone"""
    errors_gps, errors_dgps = [], []
    for seed in range(30):
        sim = GPSSimulator(seed=seed)
        result = sim.run()
        errors_gps.append(result.true_position.distance_to(result.gps_position))
        errors_dgps.append(result.true_position.distance_to(result.dgps_position))
    assert np.mean(errors_dgps) < np.mean(errors_gps)

def test_full_run():
    sim = GPSSimulator(seed=0)
    result = sim.run(n_satellites=8)
    assert result.gps_position is not None
    assert result.dgps_position is not None
    assert len(result.satellites) == 8
    assert len(result.step_log) > 0

def test_receiver_position_distance():
    a = ReceiverPosition(0, 0)
    b = ReceiverPosition(3, 4)
    assert abs(a.distance_to(b) - 5.0) < 1e-9
