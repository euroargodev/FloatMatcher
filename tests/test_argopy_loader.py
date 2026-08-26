# tests/test_argopy_loaders.py


import numpy as np
import pandas as pd
import pytest
import xarray as xr

from floatmatcher.profile_loader import ProfileLoader
from floatmatcher.exceptions import ProfileFormatError


# ─────────────────────────────────────────────────────────────
#  from_argopy_index  (a pandas DataFrame, one row per profile)
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def argopy_index_df():
    """DataFrame with argopy's real index columns (file/date/longitude/...)."""
    return pd.DataFrame({
        "file": ["D6902746_001.nc", "D6902746_002.nc", "D6902746_003.nc"],
        "date": pd.to_datetime(["2017-07-06 14:49", "2017-07-08 06:49",
                                "2017-07-13 06:53"]),
        "longitude": [-60.173, -60.140, -59.992],
        "latitude": [20.079, 20.130, 20.212],
        "wmo": [6902746, 6902746, 6902746],
    })


def test_from_argopy_index_reads_lonlatdate(argopy_index_df):
    ps = ProfileLoader.from_argopy_index(argopy_index_df)
    assert len(ps.lon) == 3
    np.testing.assert_allclose(ps.lon, [-60.173, -60.140, -59.992])
    np.testing.assert_allclose(ps.lat, [20.079, 20.130, 20.212])
    assert ps.time.dtype == np.dtype("datetime64[ns]")


def test_from_argopy_index_carries_row_provenance(argopy_index_df):
    ps = ProfileLoader.from_argopy_index(argopy_index_df)
    # default RangeIndex -> origin_dim "index", origin_index the row numbers
    assert ps.origin_dim == "index"
    np.testing.assert_array_equal(np.asarray(ps.origin_index), np.arange(3))


# ─────────────────────────────────────────────────────────────
#  from_argopy_float, N_PROF layout (no accessor needed)
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def argopy_float_profiles():
    """Profile layout (N_PROF, N_LEVELS): one lon/lat/time per profile."""
    return xr.Dataset(
        {"TEMP": (("N_PROF", "N_LEVELS"), np.zeros((3, 5)))},
        coords={
            "N_PROF": np.arange(3),
            "N_LEVELS": np.arange(5),
            "LONGITUDE": ("N_PROF", np.array([-60.173, -60.140, -59.992])),
            "LATITUDE": ("N_PROF", np.array([20.079, 20.130, 20.212])),
            "TIME": ("N_PROF", np.array(
                ["2017-07-06T14:49", "2017-07-08T06:49", "2017-07-13T06:53"],
                dtype="datetime64[ns]")),
        },
    )


def test_from_argopy_float_profile_layout(argopy_float_profiles):
    ps = ProfileLoader.from_argopy_float(argopy_float_profiles)
    assert len(ps.lon) == 3
    assert ps.origin_dim == "N_PROF"
    np.testing.assert_array_equal(np.asarray(ps.origin_index), np.arange(3))


# ─────────────────────────────────────────────────────────────
#  from_argopy_float, N_POINTS layout -> deduplication to profiles
# ─────────────────────────────────────────────────────────────

class _FakeArgoAccessor:
    """Stand-in for argopy's 'argo' accessor: only point2profile() is needed.
    Collapses repeated measurement rows to one row per profile (keyed by TIME)."""

    def __init__(self, ds):
        self._ds = ds

    def point2profile(self):
        d = self._ds
        _, first = np.unique(d["TIME"].values, return_index=True)
        first = np.sort(first)
        return xr.Dataset(
            {"TEMP": (("N_PROF", "N_LEVELS"), np.zeros((len(first), 5)))},
            coords={
                "N_PROF": np.arange(len(first)),
                "N_LEVELS": np.arange(5),
                "LONGITUDE": ("N_PROF", d["LONGITUDE"].values[first]),
                "LATITUDE": ("N_PROF", d["LATITUDE"].values[first]),
                "TIME": ("N_PROF", d["TIME"].values[first]),
            },
        )


@pytest.fixture
def fake_argo_accessor():
    """Register the fake 'argo' accessor for the duration of a test."""
    xr.register_dataset_accessor("argo")(_FakeArgoAccessor)
    yield
    # xarray has no public de-register; overwriting on re-register is fine for tests


@pytest.fixture
def argopy_float_measurements():
    """Measurement layout (N_POINTS): 3 profiles x 4 levels, lon/lat/time
    repeated within each profile."""
    lon = np.repeat([-60.173, -60.140, -59.992], 4)
    lat = np.repeat([20.079, 20.130, 20.212], 4)
    tim = np.repeat(np.array(
        ["2017-07-06T14:49", "2017-07-08T06:49", "2017-07-13T06:53"],
        dtype="datetime64[ns]"), 4)
    return xr.Dataset(
        {"TEMP": (("N_POINTS",), np.zeros(12))},
        coords={"N_POINTS": np.arange(12), "LONGITUDE": ("N_POINTS", lon),
                "LATITUDE": ("N_POINTS", lat), "TIME": ("N_POINTS", tim)},
    )


def test_from_argopy_float_deduplicates_measurements(
        fake_argo_accessor, argopy_float_measurements):
    ps = ProfileLoader.from_argopy_float(argopy_float_measurements)
    # 12 measurement rows -> 3 profile positions
    assert len(ps.lon) == 3
    np.testing.assert_allclose(ps.lon, [-60.173, -60.140, -59.992])
    assert ps.origin_dim == "N_PROF"


def test_from_argopy_float_missing_accessor_raises(argopy_float_measurements):
    """An N_POINTS dataset without the argo accessor gives a clear error.
    (This test must run with NO 'argo' accessor registered.)"""
    ds = argopy_float_measurements
    # ensure no usable accessor: temporarily register one that lacks point2profile
    class _NoP2P:
        def __init__(self, ds): pass
    xr.register_dataset_accessor("argo")(_NoP2P)
    with pytest.raises(ProfileFormatError):
        ProfileLoader.from_argopy_float(ds)