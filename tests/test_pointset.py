# tests/test_pointset.py
import numpy as np
import numpy.testing as npt
import pytest

from floatmatcher.pointset import PointSet


def _times(n):
    """Helper: n consecutive daily timestamps."""
    return np.array([np.datetime64("2015-01-01") + np.timedelta64(i, "D")
                     for i in range(n)])


# ───────────── validation ─────────────

def test_time_dtype():
    """test dtype of time array"""
    ps = PointSet([-45.0, -44.0], [32.0, 33.0], _times(2))
    assert ps.time.dtype == "datetime64[ns]"

def test_valid_lengths_ok():
    """Matching lengths build without error."""
    ps = PointSet([-45.0, -44.0], [32.0, 33.0], _times(2))
    assert len(ps.lon) == 2


def test_mismatched_lengths_raise():
    """Different lengths are rejected at construction."""
    with pytest.raises(ValueError):
        PointSet([-45.0, -44.0], [32.0], _times(2))   # lat too short


def test_arrays_are_converted():
    """Python lists become float NumPy arrays after construction."""
    ps = PointSet([1, 2, 3], [4, 5, 6], _times(3))    # ints on purpose
    assert isinstance(ps.lon, np.ndarray)
    assert ps.lon.dtype == float


# ───────────── xyz cache ─────────────

def test_xyz_shape():
    """xyz exposes one (x, y, z) row per point."""
    ps = PointSet([0.0, 90.0, 0.0], [0.0, 0.0, 90.0], _times(3))
    assert ps.xyz.shape == (3, 3)


def test_xyz_is_cached():
    """xyz is computed once and the same array is returned afterwards."""
    ps = PointSet([10.0, 20.0], [30.0, 40.0], _times(2))
    first = ps.xyz
    second = ps.xyz
    assert first is second          # same object, not just equal → cache hit


def test_xyz_matches_geo():
    """xyz matches the underlying geo conversion."""
    from floatmatcher.geo import lonlat_to_xyz
    lon = [0.0, 45.0]
    lat = [0.0, 45.0]
    ps = PointSet(lon, lat, _times(2))
    npt.assert_allclose(ps.xyz, lonlat_to_xyz(lon, lat))



# ───────────── provenance ─────────────

def test_provenance_defaults_to_none():
    """Without a source, provenance fields are None."""
    ps = PointSet([1.0], [2.0], _times(1))
    assert ps.origin_index is None
    assert ps.origin_dim is None


def test_provenance_is_kept():
    """Provenance is stored when provided."""
    idx = np.array([10, 11, 12])
    ps = PointSet([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], _times(3),
                  origin_index=idx, origin_dim="N_POINTS")
    npt.assert_array_equal(ps.origin_index, idx)
    assert ps.origin_dim == "N_POINTS"


# ───────────── test longitude convention ─────────────

def test_lon_in_converts_to_360():
    ps = PointSet([-45.0, -10.0, 0.0], [0.0, 0.0, 0.0], _times(3))
    npt.assert_allclose(ps.lon_in("0-360"), [315.0, 350.0, 0.0])
 
 
def test_lon_in_converts_to_180():
    ps = PointSet([200.0, 350.0], [0.0, 0.0], _times(2))
    npt.assert_allclose(ps.lon_in("-180-180"), [-160.0, -10.0])
 
 
def test_lon_in_does_not_mutate_self():
    # the stored longitudes must stay in the user's original convention
    ps = PointSet([200.0, 350.0], [0.0, 0.0], _times(2))
    _ = ps.lon_in("-180-180")
    npt.assert_allclose(ps.lon, [200.0, 350.0])
 
 
def test_lon_in_is_cached():
    ps = PointSet([200.0, 350.0], [0.0, 0.0], _times(2))
    first = ps.lon_in("-180-180")
    second = ps.lon_in("-180-180")
    assert first is second          # same object -> cache hit
 
 
def test_lon_in_different_ranges_cached_separately():
    ps = PointSet([200.0], [0.0], _times(1))
    a = ps.lon_in("0-360")
    b = ps.lon_in("-180-180")
    assert a is not b               # each convention cached independently
 
 
def test_lon_in_preserves_order():
    # conversion must not sort (keeps alignment with origin_index)
    ps = PointSet([350.0, 10.0, 200.0], [0.0, 0.0, 0.0], _times(3))
    npt.assert_allclose(ps.lon_in("-180-180"), [-10.0, 10.0, -160.0])
 
