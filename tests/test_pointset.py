# tests/test_pointset.py
import warnings

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr

from floatmatcher.exceptions import ProfileFormatError
from floatmatcher.pointset import PointSet
from floatmatcher.utils import extract, find_key, get

from helpers import daily_timestamps


# ---------- validation ----------
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


# ---------- xyz cache ----------

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



# ---------- provenance ----------

def test_provenance_defaults_to_none():
    """Without a source, provenance fields are None."""
    ps = PointSet([1.0], [2.0], daily_timestamps(1))
    assert ps.points_dim is None


def test_time_is_optional():
    ps = PointSet([1.0, 2.0], [3.0, 4.0])
    assert ps.time is None
    assert len(ps.lon) == 2


# ---------- Loaders fixtures (former ProfileLoader tests) ----------
@pytest.fixture
def argopy_like_ds():
    """xarray Dataset in the argopy N_PROF layout, with a real coordinate
    on the point dimension -> exercises label provenance."""
    return xr.Dataset(
        {
            "LONGITUDE": (("N_PROF",), np.array([-45.0, -44.0, -43.0])),
            "LATITUDE": (("N_PROF",), np.array([32.0, 33.0, 34.0])),
            "TIME": (("N_PROF",), np.array(
                ["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]")),
        },
        coords={"N_PROF": np.arange(3)},
    )


@pytest.fixture
def juld_ds():
    """xarray Dataset whose time variable is named JULD, not TIME
    -> exercises the TIME/JULD tolerance."""
    return xr.Dataset(
        {
            "LONGITUDE": (("obs",), np.array([-45.0, -44.0, -43.0])),
            "LATITUDE": (("obs",), np.array([32.0, 33.0, 34.0])),
            "JULD": (("obs",), np.array(
                ["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]")),
        },
    )


@pytest.fixture
def df_datetime_index():
    """DataFrame whose index is a DatetimeIndex (non-integer)"""
    idx = pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03"])
    idx.name = "profile_date"
    return pd.DataFrame(
        {
            "longitude": [-45.0, -44.0, -43.0],
            "latitude": [32.0, 33.0, 34.0],
            "date": idx,
        },
        index=idx,
    )


# ---------- Helpers: find_key / get / extract ----------
def test_find_key_matches_variable_or_coord(argopy_like_ds):
    # variable match
    assert find_key(argopy_like_ds, "LONGITUDE") == "LONGITUDE"
    # coordinate match (N_PROF is a coord here)
    assert find_key(argopy_like_ds, "N_PROF") == "N_PROF"


def test_find_key_raises_when_no_candidate_matches(argopy_like_ds):
    with pytest.raises(ProfileFormatError):
        find_key(argopy_like_ds, "DUMMY")


def test_get_returns_dataarray_extract_returns_ndarray(argopy_like_ds):
    da = get(argopy_like_ds, "LONGITUDE")
    assert isinstance(da, xr.DataArray)      # object, has .dims
    arr = extract(argopy_like_ds, "LONGITUDE")
    assert isinstance(arr, np.ndarray)       # raw values


# ---------- from_arrays ----------
def test_from_arrays_builds_without_provenance():
    ps = PointSet.from_arrays(
        lon=[-45.0, -44.0],
        lat=[32.0, 33.0],
        time=np.array(["2015-01-01", "2015-01-02"], dtype="datetime64[ns]"),
    )
    npt.assert_allclose(ps.lon, [-45.0, -44.0])
    assert ps.points_dim is None
    assert ps.original_data is None


def test_from_arrays_length_mismatch_raises():
    with pytest.raises(ValueError):
        PointSet.from_arrays(
            lon=[-45.0, -44.0, -43.0],
            lat=[32.0, 33.0],                # shorter on purpose
            time=np.array(["2015-01-01", "2015-01-02"], dtype="datetime64[ns]"),
        )


def test_from_arrays_without_time():
    ps = PointSet.from_arrays(lon=[-45.0, -44.0], lat=[32.0, 33.0])
    assert ps.time is None


# ---------- from_dataframe ----------
def test_from_dataframe_uses_the_index_name(df_datetime_index):
    """points_dim comes from df.index.name"""
    ps = PointSet.from_dataframe(df_datetime_index)

    assert ps.points_dim == "profile_date"
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
    ps = PointSet.from_dataframe(df)
    assert ps.points_dim == "index"


def test_from_dataframe_carries_the_original_dataframe(df_datetime_index):
    ps = PointSet.from_dataframe(df_datetime_index)

    assert ps.original_data is df_datetime_index
    assert ps.points_dim == "profile_date"


def test_from_dataframe_without_time_column_warns(df_datetime_index):
    with pytest.warns(UserWarning, match="no time"):
        ps = PointSet.from_dataframe(df_datetime_index.drop(columns="date"))
    assert ps.time is None


def test_from_dataframe_missing_lon_or_lat_raises(df_datetime_index):
    with pytest.raises(ProfileFormatError, match="latitude"):
        PointSet.from_dataframe(df_datetime_index.drop(columns="latitude"))
    with pytest.raises(ProfileFormatError, match="available"):
        PointSet.from_dataframe(df_datetime_index, lon="lon")       # wrong name given


def test_from_dataframe_time_none(df_datetime_index):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # time=None is explicit: no warning
        ps = PointSet.from_dataframe(df_datetime_index.drop(columns="date"), time=None)
    assert ps.time is None


def test_from_dataframe_single_row_series(df_datetime_index):
    """One row (pd.Series) gives scalars -> 1-element arrays, origin kept as a DataFrame."""
    ps = PointSet.from_dataframe(df_datetime_index.iloc[0])

    npt.assert_allclose(ps.lon, [-45.0])
    npt.assert_allclose(ps.lat, [32.0])
    assert len(ps.time) == 1
    assert isinstance(ps.original_data, pd.DataFrame)
    assert len(ps.original_data) == 1


# ---------- from_xrdataset ----------

def test_from_xrdataset_time_juld_tolerance(juld_ds):
    # time defaults to "TIME"; JULD must still be found as a fallback.
    ps = PointSet.from_xrdataset(juld_ds)
    npt.assert_array_equal(ps.time,
                           np.array(["2015-01-01", "2015-01-02", "2015-01-03"],
                                    dtype="datetime64[ns]"))


def test_from_xrdataset_without_time_variable():
    ds = xr.Dataset({
        "LONGITUDE": (("obs",), np.array([-45.0, -44.0])),
        "LATITUDE": (("obs",), np.array([32.0, 33.0])),
    })
    with pytest.warns(UserWarning, match="no time"):
        ps = PointSet.from_xrdataset(ds)
    assert ps.time is None
    assert ps.points_dim == "obs"


def test_from_xrdataset_time_none(argopy_like_ds):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # time=None is explicit: no warning
        ps = PointSet.from_xrdataset(argopy_like_ds, time=None)
    assert ps.time is None


def test_from_xrdataset_user_can_override_names():
    ds = xr.Dataset({
        "my_lon": (("obs",), np.array([-45.0, -44.0])),
        "my_lat": (("obs",), np.array([32.0, 33.0])),
        "my_time": (("obs",), np.array(
            ["2015-01-01", "2015-01-02"], dtype="datetime64[ns]")),
    })
    ps = PointSet.from_xrdataset(ds, lon="my_lon", lat="my_lat", time="my_time")
    npt.assert_allclose(ps.lon, [-45.0, -44.0])


def test_from_xrdataset_missing_coordinate_raises():
    ds = xr.Dataset({
        "LONGITUDE": (("obs",), np.array([-45.0, -44.0])),
        # LATITUDE missing on purpose
        "TIME": (("obs",), np.array(
            ["2015-01-01", "2015-01-02"], dtype="datetime64[ns]")),
    })
    with pytest.raises(ProfileFormatError):
        PointSet.from_xrdataset(ds)


def test_from_xrdataset_carries_the_original_dataset(argopy_like_ds):
    ps = PointSet.from_xrdataset(argopy_like_ds)

    assert ps.original_data is argopy_like_ds
    assert ps.points_dim == "N_PROF"
