# tests/test_nearest.py
import numpy as np
import numpy.testing as npt

from floatmatcher.nearest import NearestNeighbor
from floatmatcher.gridset import GridSet
from floatmatcher.pointset import PointSet
from floatmatcher.method import Constraints


def set_timestamps(*days):
    """Timestamps for the given 2015-01-xx days."""
    return np.array([f"2015-01-{d:02d}" for d in days], dtype="datetime64[ns]")

# ─────────────────────────────────────────────────────────────
# --- 2D: picks the nearest node's value ---
# ─────────────────────────────────────────────────────────────

def test_2d_picks_nearest_node(standard_grid_2d):
    """A point near a node gets that node's (position-encoded) value."""
    grid = GridSet(standard_grid_2d)
    # points slightly off exact nodes → nearest node is the rounded one
    lon = np.array([-49.9, -47.1])
    lat = np.array([31.1, 30.0])
    points = PointSet(lon, lat, set_timestamps(1, 1))

    result = NearestNeighbor().match(grid, points, Constraints())

    # nearest nodes: (lat=31, lon=-50) and (lat=30, lon=-47)
    expected = np.array([1000 * 31 + (-50), 1000 * 30 + (-47)])
    npt.assert_allclose(result.values["v"], expected)
    assert result.valid.all()

# ─────────────────────────────────────────────────────────────
# --- 3D: picks the nearest node in space AND time ---
# ─────────────────────────────────────────────────────────────

def test_3d_picks_nearest_in_space_and_time(standard_grid_3d):
    """The retained node is nearest in space and in time."""
    grid = GridSet(standard_grid_3d)
    lon = np.array([-49.9])
    lat = np.array([31.1])
    time = set_timestamps(3)                         # closest to the 3rd time step (index 2)
    points = PointSet(lon, lat, time)

    result = NearestNeighbor().match(grid, points, Constraints())

    # node (lat=31, lon=-50), time index 2 → 1000*31 - 50 + 100000*2
    expected = np.array([1000 * 31 + (-50) + 100000 * 2])
    npt.assert_allclose(result.values["v"], expected)

# ─────────────────────────────────────────────────────────────
# --- Thresholds in space AND time ---
# ─────────────────────────────────────────────────────────────

def test_distance_constraint_is_inclusive(standard_grid_2d):
    """A point exactly at the threshold distance is accepted (<=)."""
    grid = GridSet(standard_grid_2d)
    lon = np.array([-46.3])              
    lat = np.array([34.2])
    points = PointSet(lon, lat, set_timestamps(1))

    # observe the exact distance the engine computes for this point
    observed = NearestNeighbor().match(grid, points, Constraints(max_dist_km=1e9))
    d = observed.distance_km[0]

    # threshold set EXACTLY at that distance → inclusive policy accepts it
    at_threshold = NearestNeighbor().match(grid, points, Constraints(max_dist_km=d))
    assert at_threshold.valid[0]                       

    # threshold a hair below → rejected (confirms the boundary is real)
    below = NearestNeighbor().match(grid, points, Constraints(max_dist_km=d - 1e-6))
    assert not below.valid[0]
    assert np.isnan(below.values["v"][0])


def test_time_constraint_is_inclusive(standard_grid_3d):
    """A point exactly at the threshold time delta is accepted (<=)."""
    grid = GridSet(standard_grid_3d)
    print(grid)
    lon = np.array([-48.0])
    lat = np.array([31.0])
    # a time between grid steps so the delta is non-zero
    points = PointSet(lon, lat, np.array(["2015-01-03T06:00"], dtype="datetime64[ns]"))

    dt=0.25     # correspond to 6houres shift of points

    at_threshold = NearestNeighbor().match(grid, points, Constraints(max_time_days=dt))
    assert at_threshold.valid[0]                       

    below = NearestNeighbor().match(grid, points, Constraints(max_time_days=dt - 1e-9))
    assert not below.valid[0]

# ─────────────────────────────────────────────────────────────
# --- constraints: mask, not filter ---
# ─────────────────────────────────────────────────────────────

def test_distance_constraint_invalidates_far_point(standard_grid_2d):
    """A point beyond max_dist_km is invalid and NaN, but still present."""
    grid = GridSet(standard_grid_2d)
    lon = np.array([-49.0, 0.0])          # second point is ~thousands of km away
    lat = np.array([31.0, 31.0])
    points = PointSet(lon, lat, set_timestamps(1, 1))

    result = NearestNeighbor().match(grid, points, Constraints(max_dist_km=50))

    assert result.valid[0]
    assert not result.valid[1]
    assert np.isnan(result.values["v"][1])       # masked, not dropped
    assert len(result.values["v"]) == 2          # length preserved
    npt.assert_allclose(result.distance_km[1], np.nan)  # invalid → NaN distance


def test_time_constraint_invalidates_far_in_time(standard_grid_3d):
    """A point too far in time is invalidated, even if spatially close."""
    grid = GridSet(standard_grid_3d)
    lon = np.array([-49.0])
    lat = np.array([31.0])
    time = set_timestamps(20)                         
    points = PointSet(lon, lat, time)

    result = NearestNeighbor().match(grid, points, Constraints(max_time_days=1))

    assert not result.valid[0]
    assert np.isnan(result.values["v"][0])

# ─────────────────────────────────────────────────────────────
# --- output alignment ---
# ─────────────────────────────────────────────────────────────

def test_result_is_aligned_with_points(standard_grid_2d):
    """Every result array has one entry per input point, in order."""
    grid = GridSet(standard_grid_2d)
    lon = np.array([-49.0, -48.0, -47.0])
    lat = np.array([30.0, 31.0, 32.0])
    points = PointSet(lon, lat, set_timestamps(1, 1, 1))

    result = NearestNeighbor().match(grid, points, Constraints())

    assert len(result.valid) == 3
    assert len(result.distance_km) == 3
    assert len(result.values["v"]) == 3