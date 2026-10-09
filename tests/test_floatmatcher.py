# tests/test_FloatMatcher.py
#

import numpy as np
import numpy.testing as npt
import pytest

from floatmatcher.methods import NearestNeighbor
from floatmatcher.floatmatcher import FloatMatcher
from floatmatcher.pointset import PointSet
from floatmatcher.product import ERA5Product, LUTProduct


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

def test_set_points_from_arrays():
    fm = FloatMatcher()
    fm.set_points_from_arrays([11.1, 9.05], [30.05, 39.95],
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


# ---------- builder : match through the set_* ----------

def test_match_through_the_builder(era5_files):
    fm = FloatMatcher()
    fm.set_points_from_arrays([11.1], [30.05], np.array(["2015-06-01T02"], dtype="datetime64[ns]"))
    fm.set_product("era5", "local", era5_files, ["sst"])
    fm.set_method("nearest")

    fm.match()
    res = fm.result

    assert res.valid.tolist() == [True]
    npt.assert_allclose(res.values["sst"], [121.03], atol=1e-4)


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
