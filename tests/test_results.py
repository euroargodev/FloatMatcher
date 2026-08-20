# tests/test_result.py
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from floatmatcher.results import MatchupResult
from floatmatcher.pointset import PointSet


def _result(n=3):
    """A minimal 3-point result with one variable."""
    return MatchupResult(
        values={"t2m": np.array([280.0, 281.0, np.nan])},
        distance_km=np.array([1.0, 2.0, np.nan]),
        time_delta=np.array([0.1, 0.2, np.nan]),
        valid=np.array([True, True, False]),
    )


# --- structure ---
def test_fields_are_aligned():
    """All arrays share the point count; values is keyed by variable."""
    r = _result()
    assert len(r.valid) == 3
    assert len(r.distance_km) == len(r.valid)
    assert set(r.values) == {"t2m"}


# --- to_dataset: nominal ---
def test_to_dataset_adds_coloc_variable(points_with_origin):
    """Reinjection adds a '<var>_coloc' variable on the origin dimension."""
    ds = xr.Dataset(coords={"N_POINTS": np.arange(3)})
    out = _result().to_dataset(ds, points_with_origin)

    assert "t2m_coloc" in out.data_vars
    assert out["t2m_coloc"].dims == ("N_POINTS",)
    npt.assert_array_equal(out["t2m_coloc"].values,
                           np.array([280.0, 281.0, np.nan]))


def test_to_dataset_does_not_mutate_original(points_with_origin):
    """The original dataset is left untouched (we work on a copy)."""
    ds = xr.Dataset(coords={"N_POINTS": np.arange(3)})
    _result().to_dataset(ds, points_with_origin)
    assert "t2m_coloc" not in ds.data_vars      # original still clean


# --- to_dataset: guard rails ---
def test_to_dataset_without_provenance_raises():
    """Raw-array points (no origin_dim) cannot be reinjected."""
    points = PointSet([-45.0, -44.0, -43.0], [32.0, 33.0, 34.0],
                      np.array(["2015-01-01", "2015-01-02", "2015-01-03"],
                               dtype="datetime64[ns]"))
    ds = xr.Dataset(coords={"N_POINTS": np.arange(3)})
    with pytest.raises(ValueError):
        _result().to_dataset(ds, points)        # origin_dim is None


def test_to_dataset_length_mismatch_raises(points_with_origin):
    """A dataset of the wrong length is rejected."""
    ds = xr.Dataset(coords={"N_POINTS": np.arange(5)})   # 5 != 3 results
    with pytest.raises(ValueError):
        _result().to_dataset(ds, points_with_origin)