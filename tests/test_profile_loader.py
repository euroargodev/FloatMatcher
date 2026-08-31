# Tests for ProfileLoader and its private extraction helpers.
#
# Each test targets a DESIGN DECISION we made, so a regression on that decision
# fails loudly. No I/O, no network: everything runs on in-memory objects.
#
# NOTE: fix the imports below to match your package layout.
#   - ProfileLoader / helpers module path
#   - whether ProfileFormatError is public

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

# Fixtures argopy_like_ds / raw_ds_no_coord / juld_ds / df_datetime_index
# are expected to be available (from loader_fixtures.py or a conftest).

# ─────────────────────────────────────────────────────────────
#  Helpers: _find_key / _get / _extract
# ─────────────────────────────────────────────────────────────

def test_find_key_returns_first_present_skipping_absent(argopy_like_ds):
    # "TIME" is absent as a first candidate? no -> use one we know is absent first.
    # LONGITUDE exists; a bogus name before it must be skipped, not raise.
    assert _find_key(argopy_like_ds, "NOPE", "LONGITUDE") == "LONGITUDE"


def test_find_key_matches_variable_or_coord(argopy_like_ds):
    # variable match
    assert _find_key(argopy_like_ds, "LONGITUDE") == "LONGITUDE"
    # coordinate match (N_PROF is a coord here)
    assert _find_key(argopy_like_ds, "N_PROF") == "N_PROF"


def test_find_key_raises_when_no_candidate_matches(argopy_like_ds):
    with pytest.raises(ProfileFormatError):
        _find_key(argopy_like_ds, "NOPE", "STILL_NOPE")


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
    assert len(ps.lon) == 2
    assert ps.origin_dim is None


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
#  from_xrdataset — the core decisions
# ─────────────────────────────────────────────────────────────

def test_from_xrdataset_time_juld_tolerance(juld_ds):
    # time defaults to "TIME"; JULD must still be found as a fallback.
    ps = ProfileLoader.from_xrdataset(juld_ds)
    assert len(ps.time) == 3


def test_from_xrdataset_user_can_override_names():
    ds = xr.Dataset({
        "my_lon": (("obs",), np.array([-45.0, -44.0])),
        "my_lat": (("obs",), np.array([32.0, 33.0])),
        "my_time": (("obs",), np.array(
            ["2015-01-01", "2015-01-02"], dtype="datetime64[ns]")),
    })
    ps = ProfileLoader.from_xrdataset(ds, lon="my_lon", lat="my_lat", time="my_time")
    assert len(ps.lon) == 2


def test_from_xrdataset_missing_coordinate_raises():
    ds = xr.Dataset({
        "LONGITUDE": (("obs",), np.array([-45.0, -44.0])),
        # LATITUDE missing on purpose
        "TIME": (("obs",), np.array(
            ["2015-01-01", "2015-01-02"], dtype="datetime64[ns]")),
    })
    with pytest.raises(ProfileFormatError):
        ProfileLoader.from_xrdataset(ds)