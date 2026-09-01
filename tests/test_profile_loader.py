# Tests for ProfileLoader and helpers.


import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr

from floatmatcher.profile_loader import (
    ProfileLoader,
    _find_key,
    _get,
    _extract,
)
from floatmatcher.exceptions import ProfileFormatError

# Fixtures argopy_like_ds / juld_ds / df_datetime_index
# are expected to be available (from loader_fixtures.py or a conftest).

# ─────────────────────────────────────────────────────────────
#  Helpers: _find_key / _get / _extract
# ─────────────────────────────────────────────────────────────

def test_find_key_matches_variable_or_coord(argopy_like_ds):
    # variable match
    assert _find_key(argopy_like_ds, "LONGITUDE") == "LONGITUDE"
    # coordinate match (N_PROF is a coord here)
    assert _find_key(argopy_like_ds, "N_PROF") == "N_PROF"


def test_find_key_raises_when_no_candidate_matches(argopy_like_ds):
    with pytest.raises(ProfileFormatError):
        _find_key(argopy_like_ds, "DUMMY")


def test_get_returns_dataarray_extract_returns_ndarray(argopy_like_ds):
    da = _get(argopy_like_ds, "LONGITUDE")
    assert isinstance(da, xr.DataArray)      # object, has .dims
    arr = _extract(argopy_like_ds, "LONGITUDE")
    assert isinstance(arr, np.ndarray)       # raw values


# ─────────────────────────────────────────────────────────────
#  from_arrays
# ─────────────────────────────────────────────────────────────

def test_from_arrays_builds_without_provenance():
    ps = ProfileLoader.from_arrays(
        lon=[-45.0, -44.0],
        lat=[32.0, 33.0],
        time=np.array(["2015-01-01", "2015-01-02"], dtype="datetime64[ns]"),
    )
    npt.assert_allclose(ps.lon, [-45.0, -44.0])
    assert ps.origin_dim is None
    assert ps.origin_ds is None


def test_from_arrays_length_mismatch_raises():
    # PointSet enforces equal lengths; the loader should surface that.
    with pytest.raises(ValueError):
        ProfileLoader.from_arrays(
            lon=[-45.0, -44.0, -43.0],
            lat=[32.0, 33.0],                # shorter on purpose
            time=np.array(["2015-01-01", "2015-01-02"], dtype="datetime64[ns]"),
        )


# ─────────────────────────────────────────────────────────────
#  from_dataframe
# ─────────────────────────────────────────────────────────────

def test_from_dataframe_uses_the_index_name(df_datetime_index):
    """origin_dim comes from df.index.name"""
    ps = ProfileLoader.from_dataframe(df_datetime_index)

    assert ps.origin_dim == "profile_date"
    npt.assert_allclose(ps.lon, [-45.0, -44.0, -43.0])
    npt.assert_allclose(ps.lat, [32.0, 33.0, 34.0])
    npt.assert_array_equal(ps.time,
                           np.array(["2015-01-01", "2015-01-02", "2015-01-03"],
                                    dtype="datetime64[ns]"))


def test_from_dataframe_unnamed_index_falls_back_to_index():
    df = pd.DataFrame({
        "longitude": [-45.0, -44.0],
        "latitude": [32.0, 33.0],
        "date": pd.to_datetime(["2015-01-01", "2015-01-02"]),
    })  # default RangeIndex, name is None
    ps = ProfileLoader.from_dataframe(df)
    assert ps.origin_dim == "index"


# ─────────────────────────────────────────────────────────────
#  from_xrdataset
# ─────────────────────────────────────────────────────────────

def test_from_xrdataset_time_juld_tolerance(juld_ds):
    # time defaults to "TIME"; JULD must still be found as a fallback.
    ps = ProfileLoader.from_xrdataset(juld_ds)
    npt.assert_array_equal(ps.time,
                           np.array(["2015-01-01", "2015-01-02", "2015-01-03"],
                                    dtype="datetime64[ns]"))


def test_from_xrdataset_user_can_override_names():
    ds = xr.Dataset({
        "my_lon": (("obs",), np.array([-45.0, -44.0])),
        "my_lat": (("obs",), np.array([32.0, 33.0])),
        "my_time": (("obs",), np.array(
            ["2015-01-01", "2015-01-02"], dtype="datetime64[ns]")),
    })
    ps = ProfileLoader.from_xrdataset(ds, lon="my_lon", lat="my_lat", time="my_time")
    npt.assert_allclose(ps.lon, [-45.0, -44.0])


def test_from_xrdataset_missing_coordinate_raises():
    ds = xr.Dataset({
        "LONGITUDE": (("obs",), np.array([-45.0, -44.0])),
        # LATITUDE missing on purpose
        "TIME": (("obs",), np.array(
            ["2015-01-01", "2015-01-02"], dtype="datetime64[ns]")),
    })
    with pytest.raises(ProfileFormatError):
        ProfileLoader.from_xrdataset(ds)


# ───────────── origin_ds: the provenance the reinjection relies on ─────────────

def test_from_xrdataset_carries_the_original_dataset(argopy_like_ds):
    ps = ProfileLoader.from_xrdataset(argopy_like_ds)

    assert ps.origin_ds is argopy_like_ds
    assert ps.origin_dim == "N_PROF"


def test_from_dataframe_has_no_original_dataset(df_datetime_index):
    """A DataFrame is not a Dataset: origin_dim is known, origin_ds is not."""
    ps = ProfileLoader.from_dataframe(df_datetime_index)

    assert ps.origin_ds is None
    assert ps.origin_dim == "profile_date"
