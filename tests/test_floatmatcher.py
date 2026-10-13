# tests/test_FloatMatcher.py
#

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr

from floatmatcher.methods import NearestNeighbor
from floatmatcher.floatmatcher import FloatMatcher
from floatmatcher.pointset import PointSet
from floatmatcher.product import ERA5Product, LUTProduct


# times of the two points used with the grid_3d_ds file (t0 + 2 h, t1 - 1 h)
TIMES = np.array(["2015-01-01T02", "2015-01-01T23"], dtype="datetime64[ns]")


def _matched(grid_3d_ds, grid_file, *data, **names):
    """FloatMatcher with the given points, matched on the grid_3d_ds file."""
    fm = FloatMatcher()
    fm.set_points(*data, **names)
    fm.set_product("era5", "local", grid_file(grid_3d_ds), ["sst"])
    fm.set_method("nearest")
    fm.match()
    return fm


def test_match_nearest_get_good_node(era5_files):
    points = PointSet.from_arrays(
        lon=[11.1, 9.05, 349.9],
        lat=[30.05, 39.95, 20.1],
        time=np.array(["2015-06-01T02", "2015-06-02T00", "2015-06-02T22"], dtype="datetime64[ns]"),
    )
    product = ERA5Product("local", era5_files, ["sst"])

    fm = FloatMatcher(points=points, product=product, method=NearestNeighbor())
    fm.match()
    res = fm.result

    assert res.valid.all()
    assert set(res.values) == {"sst"} # only sst has been taken into account
    npt.assert_allclose(res.values["sst"], [121.03, 1.23, 232.23], atol=1e-4)
    assert (res.distance_km < 25).all() 
    assert (res.time_delta < 5 * 3600).all()          # seconds - 5h
    assert len(np.unique(res.distance_km)) == 3


def test_match_nearest_rejects_out_of_range_points(era5_files):
    points = PointSet.from_arrays(
        lon=[11.1,    230, 11.1],
        lat=[30.05, -40.0, 30.05],
        time=np.array(["2015-06-01T02",      # p0 in range
                       "2015-06-01T02",      # p1 far away in space
                       "2015-09-01T02"],     # p2 far away in time
                      dtype="datetime64[ns]"),
    )
    product = ERA5Product("local", era5_files, ["sst"])
    orch = FloatMatcher(points=points, product=product)
    orch.set_method("nearest")

    orch.match()
    res = orch.result

    assert res.valid.tolist() == [True, False, False]
    npt.assert_allclose(res.values["sst"][:1], [121.03], atol=1e-4)
    assert np.isnan(res.values["sst"][1:]).all()      # no value for the rejected
    assert np.isnan(res.distance_km[1:]).all()        # nor distance
    assert np.isnan(res.time_delta[1:]).all()         # nor time gap

    # p1 was out on DISTANCE only
    orch.set_method("nearest", radius=100000)
    orch.match()
    loose_dist = orch.result
    assert loose_dist.valid.tolist() == [True, True, False]

    # p2 was out on TIME only
    orch.set_method("nearest", time_gap=np.timedelta64(300, "D"))
    orch.match()
    loose_time = orch.result
    assert loose_time.valid.tolist() == [True, False, True]


def test_match_nearest_over_two_variable(era5_files):
    points = PointSet.from_arrays(
        lon=[11.1], lat=[30.05],
        time=np.array(["2015-06-01T02"], dtype="datetime64[ns]"),
    )
    product = ERA5Product("local", era5_files, ["sst", "t2m"])

    fm = FloatMatcher(points=points, product=product, method=NearestNeighbor())
    fm.match()
    res = fm.result

    assert set(res.values) == {"sst", "t2m"}
    npt.assert_allclose(res.values["sst"], [121.03], atol=1e-4)
    npt.assert_allclose(res.values["t2m"], [1121.03], atol=1e-4)


# ---------- builder : set_points / set_product / set_method ----------

def test_set_points():
    fm = FloatMatcher()
    fm.set_points([11.1, 9.05], [30.05, 39.95],
                              np.array(["2015-06-01T02", "2015-06-02T00"], dtype="datetime64[ns]"))

    assert isinstance(fm.points, PointSet)
    npt.assert_allclose(fm.points.lon, [11.1, 9.05])
    npt.assert_allclose(fm.points.lat, [30.05, 39.95])
    assert len(fm.points.time) == 2


def test_set_product_builds_the_product_of_the_given_type(era5_files):
    fm = FloatMatcher()

    fm.set_product("era5", "local", era5_files, ["sst"])
    assert isinstance(fm.product, ERA5Product)

    fm.set_product("lut", "local", era5_files, None)
    assert isinstance(fm.product, LUTProduct)


def test_set_product_unknown_type_raises(era5_files):
    with pytest.raises(ValueError, match="unknown product"):
        FloatMatcher().set_product("foo", "local", era5_files, None)


def test_set_product_unknown_source_param_raises(era5_files):
    """source parameters go to the source: an unknown one is refused"""
    with pytest.raises(TypeError):
        FloatMatcher().set_product("era5", "local", era5_files, None, foo=1)


def test_set_method_builds_the_method_with_its_params():
    fm = FloatMatcher()
    fm.set_method("nearest", radius=50, time_gap=np.timedelta64(3, "h"))

    assert isinstance(fm.method, NearestNeighbor)
    assert fm.method.radius == 50
    assert fm.method.time_gap_seconds == 3 * 3600


def test_set_method_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown method"):
        FloatMatcher().set_method("dummy_method")


def test_set_method_unknown_param_raises():
    with pytest.raises(TypeError):
        FloatMatcher().set_method("nearest", dummy_param=50)


# ---------- builder : set_points input types ----------

def test_set_points_without_time():
    fm = FloatMatcher()
    fm.set_points([11.1, 9.05], [30.05, 39.95])

    assert fm.points.time is None


def test_set_points_from_dataset():
    ds = xr.Dataset({"LONGITUDE": ("N_PROF", [20.1, 10.05]),
                     "LATITUDE": ("N_PROF", [1.05, 1.95]),
                     "TIME": ("N_PROF", TIMES)})
    fm = FloatMatcher()
    fm.set_points(ds)

    npt.assert_allclose(fm.points.lon, [20.1, 10.05])
    assert fm.points.original_data is ds
    assert fm.points.points_dim == "N_PROF"


def test_set_points_from_dataset_with_names():
    ds = xr.Dataset({"x": ("obs", [20.1, 10.05]), "y": ("obs", [1.05, 1.95]), "t": ("obs", TIMES)})
    fm = FloatMatcher()
    fm.set_points(ds, lon="x", lat="y", time="t")

    npt.assert_allclose(fm.points.lat, [1.05, 1.95])
    assert fm.points.points_dim == "obs"


def test_set_points_from_dataframe():
    df = pd.DataFrame({"longitude": [20.1, 10.05], "latitude": [1.05, 1.95], "date": TIMES})
    fm = FloatMatcher()
    fm.set_points(df)

    npt.assert_allclose(fm.points.lon, [20.1, 10.05])
    assert fm.points.original_data is df


def test_set_points_from_dataframe_with_names():
    df = pd.DataFrame({"LON": [20.1, 10.05], "LAT": [1.05, 1.95], "DATE": TIMES})
    fm = FloatMatcher()
    fm.set_points(df, lon="LON", lat="LAT", time="DATE")

    npt.assert_allclose(fm.points.lat, [1.05, 1.95])
    assert len(fm.points.time) == 2


def test_set_points_from_a_pointset():
    points = PointSet.from_arrays([20.1], [1.05])
    fm = FloatMatcher()
    fm.set_points(points)

    assert fm.points is points


def test_set_points_unsupported_input_raises():
    with pytest.raises(TypeError):
        FloatMatcher().set_points("not points")
    with pytest.raises(TypeError):
        FloatMatcher().set_points()


def test_set_points_names_only_for_dataframe_or_dataset():
    with pytest.raises(TypeError):
        FloatMatcher().set_points([20.1], [1.05], lon="x")
    with pytest.raises(TypeError):
        FloatMatcher().set_points(PointSet.from_arrays([20.1], [1.05]), lon="x")


# ---------- builder : match through the set_* ----------

def test_match_through_the_builder(era5_files):
    fm = FloatMatcher()
    fm.set_points([11.1], [30.05], np.array(["2015-06-01T02"], dtype="datetime64[ns]"))
    fm.set_product("era5", "local", era5_files, ["sst"])
    fm.set_method("nearest")

    fm.match()
    res = fm.result

    assert res.valid.tolist() == [True]
    npt.assert_allclose(res.values["sst"], [121.03], atol=1e-4)


# ---------- builder : reinject through the set_points input ----------
# grid_3d_ds, sst = 100*lat + lon + time_index: p0 -> node (1, 20) at t0 = 120, p1 -> node (2, 10) at t1 = 211

def test_to_dataset_through_the_builder(grid_3d_ds, grid_file):
    ds = xr.Dataset({"LONGITUDE": ("N_PROF", [20.1, 10.05]),
                     "LATITUDE": ("N_PROF", [1.05, 1.95]),
                     "TIME": ("N_PROF", TIMES)},
                    coords={"N_PROF": [10, 11]})
    fm = FloatMatcher()
    fm.set_points(ds)
    fm.set_product("era5", "local", grid_file(grid_3d_ds), ["sst"])
    fm.set_method("nearest")
    fm.match()

    out = fm.to_dataset()

    assert out["sst_coloc"].dims == ("N_PROF",)
    npt.assert_allclose(out["sst_coloc"].values, [120.0, 211.0])
    assert "sst_coloc" not in ds.data_vars            # the source is not modified


def test_to_dataframe_through_the_builder(grid_3d_ds, grid_file):
    df = pd.DataFrame({"longitude": [20.1, 10.05], "latitude": [1.05, 1.95], "date": TIMES})
    fm = FloatMatcher()
    fm.set_points(df)
    fm.set_product("era5", "local", grid_file(grid_3d_ds), ["sst"])
    fm.set_method("nearest")
    fm.match()

    out = fm.to_dataframe()

    npt.assert_allclose(out["sst_coloc"].values, [120.0, 211.0])
    assert "sst_coloc" not in df.columns              # the source is not modified


def test_reinject_before_match_raises():
    fm = FloatMatcher()
    fm.set_points([20.1], [1.05])

    with pytest.raises(ValueError, match="no result yet"):
        fm.to_dataset()
    with pytest.raises(ValueError, match="no result yet"):
        fm.to_dataframe()


def test_to_dataframe_refuses_points_from_a_dataset(grid_3d_ds, grid_file):
    ds = xr.Dataset({"LONGITUDE": ("N_PROF", [20.1, 10.05]),
                     "LATITUDE": ("N_PROF", [1.05, 1.95]),
                     "TIME": ("N_PROF", TIMES)})
    fm = _matched(grid_3d_ds, grid_file, ds)

    with pytest.raises(TypeError, match="to_dataset"):
        fm.to_dataframe()


def test_to_dataset_refuses_points_from_a_dataframe(grid_3d_ds, grid_file):
    df = pd.DataFrame({"longitude": [20.1, 10.05], "latitude": [1.05, 1.95], "date": TIMES})
    fm = _matched(grid_3d_ds, grid_file, df)

    with pytest.raises(TypeError, match="to_dataframe"):
        fm.to_dataset()


def test_reinject_refuses_points_from_arrays(grid_3d_ds, grid_file):
    """arrays carry no origin to reinject into"""
    fm = _matched(grid_3d_ds, grid_file, [20.1, 10.05], [1.05, 1.95], TIMES)

    with pytest.raises(ValueError):
        fm.to_dataset()
    with pytest.raises(ValueError):
        fm.to_dataframe()


def test_reinject_gives_a_dataset_for_points_from_a_dataset(grid_3d_ds, grid_file):
    ds = xr.Dataset({"LONGITUDE": ("N_PROF", [20.1, 10.05]),
                     "LATITUDE": ("N_PROF", [1.05, 1.95]),
                     "TIME": ("N_PROF", TIMES)})
    fm = _matched(grid_3d_ds, grid_file, ds)

    out = fm.reinject()

    assert isinstance(out, xr.Dataset)
    npt.assert_allclose(out["sst_coloc"].values, [120.0, 211.0])


def test_reinject_gives_a_dataframe_for_points_from_a_dataframe(grid_3d_ds, grid_file):
    df = pd.DataFrame({"longitude": [20.1, 10.05], "latitude": [1.05, 1.95], "date": TIMES})
    fm = _matched(grid_3d_ds, grid_file, df)

    out = fm.reinject()

    assert isinstance(out, pd.DataFrame)
    npt.assert_allclose(out["sst_coloc"].values, [120.0, 211.0])


# ---------- repr ----------

def test_repr_before_and_after_match(grid_3d_ds, grid_file):
    fm = FloatMatcher()
    assert "results: None" in repr(fm)                 # printing an empty FloatMatcher works

    fm = _matched(grid_3d_ds, grid_file, [20.1], [1.05], TIMES[:1])
    assert "nb_match_found" in repr(fm)


# ---------- match : missing inputs ----------

def _points():
    return PointSet.from_arrays([11.1], [30.05], np.array(["2015-06-01T02"], dtype="datetime64[ns]"))


def test_match_on_empty_instance_raises():
    with pytest.raises(ValueError, match="no points set"):
        FloatMatcher().match()


def test_match_without_product_raises():
    fm = FloatMatcher(points=_points(), method=NearestNeighbor())
    with pytest.raises(ValueError, match="no product set"):
        fm.match()


def test_match_without_method_raises(era5_files):
    fm = FloatMatcher(points=_points(), product=ERA5Product("local", era5_files, ["sst"]))
    with pytest.raises(ValueError, match="no method set"):
        fm.match()


# ---------- result property ----------

def test_result_before_match_raises():
    fm = FloatMatcher()

    with pytest.raises(AttributeError, match="no result yet"):
        fm.result
    assert not hasattr(fm, "result")      # attribute-like: hasattr answers instead of raising


def test_result_is_read_only():
    """the result only comes from match()"""
    fm = FloatMatcher()

    with pytest.raises(AttributeError):
        fm.result = None
