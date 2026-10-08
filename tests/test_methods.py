# tests/test_methods.py

import numpy as np
import numpy.testing as npt
import pytest

from floatmatcher.methods import Method, NearestNeighbor, available_methods
from floatmatcher.pointset import PointSet


def test_method_is_abstract():
    with pytest.raises(TypeError):
        Method()  # type: ignore[abstract]


def test_every_registered_method_is_a_method():
    assert all(issubclass(cls, Method) for cls in available_methods.values())


def test_nearest_is_registered():
    assert available_methods["nearest"] is NearestNeighbor


# ---------- NearestNeighbor parameters ----------
def test_default_tolerance_is_one_day_in_seconds():
    assert NearestNeighbor().time_gap_seconds == 86400.0


def test_time_gap_accepts_any_unit():
    assert NearestNeighbor(time_gap=np.timedelta64(3, "h")).time_gap_seconds == 10800.0
    assert NearestNeighbor(time_gap=np.timedelta64(90, "m")).time_gap_seconds == 5400.0
    assert NearestNeighbor(time_gap=np.timedelta64(2, "D")).time_gap_seconds == 172800.0


def test_unknown_param_raises():
    with pytest.raises(TypeError):
        NearestNeighbor(max_dist_km=50)  # type: ignore[call-arg]


# ---------- NearestNeighbor.apply : no FloatMatcher needed ----------
# grid_3d_ds: sst = 100*lat + lon + time_index (t2m = sst + 1000), nodes lat [0, 1, 2] x lon [10..40]
# grid_2d_ds: v = 100*lat + lon

# --> Reminder 
# 3D (grid_3d_ds): 
#        n0    n1    n2    n3     n4     n5     n6     n7     n8     n9    n10    n11
#  t=0  10.0  20.0  30.0  40.0  110.0  120.0  130.0  140.0  210.0  220.0  230.0  240.0
#  t=1  11.0  21.0  31.0  41.0  111.0  121.0  131.0  141.0  211.0  221.0  231.0  241.0

# 2D (grid_2d_ds): 
#        n0    n1    n2    n3     n4     n5     n6     n7     n8     n9    n10    n11
#  t=0  10.0  20.0  30.0  40.0  110.0  120.0  130.0  140.0  210.0  220.0  230.0  240.0



def test_apply_picks_the_nearest_node(grid_3d_ds):
    points = PointSet.from_arrays(lon=[20.0], lat=[1.0],
                                  time=np.array(["2015-01-02"], dtype="datetime64[ns]"))

    res = NearestNeighbor().apply(grid_3d_ds, points)

    assert res.valid.tolist() == [True]
    npt.assert_allclose(res.values["sst"], [121.0])
    assert res.distance_km[0] < 1e-6
    assert res.time_delta[0] == 0.0


def test_apply_rejects_out_of_radius_and_out_of_time_gap(grid_3d_ds):
    points = PointSet.from_arrays(
        lon=[20.0, 170.0, 20.0],
        lat=[1.0, 1.0, 1.0],
        time=np.array(["2015-01-02",      # p0 ok
                       "2015-01-02",      # p1 far in space
                       "2015-03-01"],     # p2 far in time
                      dtype="datetime64[ns]"),
    )

    res = NearestNeighbor().apply(grid_3d_ds, points)

    assert res.valid.tolist() == [True, False, False]
    assert np.isnan(res.values["sst"][1:]).all()
    assert np.isnan(res.distance_km[1:]).all()
    assert np.isnan(res.time_delta[1:]).all()


def test_apply_on_2d_grid_ignores_time(grid_2d_ds):
    points = PointSet.from_arrays(lon=[30.0], lat=[2.0])

    res = NearestNeighbor().apply(grid_2d_ds, points)

    assert res.valid.tolist() == [True]
    npt.assert_allclose(res.values["v"], [230.0])
    assert np.isnan(res.time_delta[0])
