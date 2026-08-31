# tests/test_results.py
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from floatmatcher.results import MatchupResult
from floatmatcher.pointset import PointSet


def _result(n=3):
    """A minimal 3-point result with one variable."""
    return MatchupResult(
        values={"dummy_variable": np.array([280.0, 281.0, np.nan])},
        distance_km=np.array([1.0, 2.0, np.nan]),
        time_delta=np.array([0.1, 0.2, np.nan]),
        valid=np.array([True, True, False]),
    )


# ---------- structure ----------
# no test structure because no __post_init__ or elements validation 


# ---------- to_dataset ----------

def test_to_dataset_adds_coloc_variable(points_with_origin):
    """Reinjection adds a '<var>_coloc' variable on the origin dimension."""
    ds = xr.Dataset(coords={"N_POINTS": np.arange(3)})
    out = _result().to_dataset(ds, points_with_origin)

    assert "dummy_variable_coloc" in out.data_vars
    assert out["dummy_variable_coloc"].dims == ("N_POINTS",)
    npt.assert_array_equal(out["dummy_variable_coloc"].values,
                           np.array([280.0, 281.0, np.nan]))


def test_to_dataset_does_not_mutate_original(points_with_origin):
    """The original dataset is left untouched (we work on a copy)."""
    ds = xr.Dataset(coords={"N_POINTS": np.arange(3)})
    _result().to_dataset(ds, points_with_origin)
    assert "dummy_variable_coloc" not in ds.data_vars      # original still clean


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