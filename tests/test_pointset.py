# tests/test_pointset.py
import numpy as np
import pytest

from floatmatcher.pointset import PointSet

from helpers import daily_timestamps


# ───────────── validation ─────────────
def test_lon_dtype():
    """test dtype of time array"""
    ps = PointSet([-45.0, -44.0], [32.0, 33.0], daily_timestamps(2))
    assert ps.lon.dtype == float

def test_lat_dtype():
    """test dtype of time array"""
    ps = PointSet([-45.0, -44.0], [32.0, 33.0], daily_timestamps(2))
    assert ps.lat.dtype == float

def test_time_dtype():
    """test dtype of time array"""
    ps = PointSet([-45.0, -44.0], [32.0, 33.0], daily_timestamps(2))
    assert ps.time.dtype == "datetime64[ns]"


def test_valid_lengths_ok():
    """Matching lengths build without error."""
    ps = PointSet([-45.0, -44.0], [32.0, 33.0], daily_timestamps(2))
    assert len(ps.lon) == len(ps.lat) == len(ps.time)


def test_mismatched_lengths_raise():
    """Different lengths are rejected at construction."""
    with pytest.raises(ValueError):
        PointSet([-45.0, -44.0], [32.0], daily_timestamps(2))   # lat too short


def test_arrays_are_converted():
    """Python lists become float NumPy arrays after construction."""
    ps = PointSet([1, 2, 3], [4, 5, 6], daily_timestamps(3))    # ints on purpose
    assert isinstance(ps.lon, np.ndarray)
    assert ps.lon.dtype == float


# ───────────── xyz cache ─────────────

def test_xyz_shape():
    """xyz exposes one (x, y, z) row per point."""
    ps = PointSet([0.0, 90.0, 0.0], [0.0, 0.0, 90.0], daily_timestamps(3))
    assert ps.xyz.shape == (3, 3)


def test_xyz_is_cached():
    """xyz is computed once and the same array is returned afterwards.
    test @property proper working """
    ps = PointSet([10.0, 20.0], [30.0, 40.0], daily_timestamps(2))
    first = ps.xyz
    second = ps.xyz
    assert first is second          # same object, not just equal → cache hit



# ───────────── provenance ─────────────

def test_provenance_defaults_to_none():
    """Without a source, provenance fields are None."""
    ps = PointSet([1.0], [2.0], daily_timestamps(1))
    assert ps.origin_dim is None


