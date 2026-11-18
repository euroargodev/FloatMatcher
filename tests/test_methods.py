# tests/test_methods.py

import numpy as np
import numpy.testing as npt
import pytest

from floatmatcher.methods import Interpolator, Method, NearestNeighbor, available_methods
from floatmatcher.pointset import PointSet
from helpers import make_grid


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
    # one neighbor in space and in time (default): one value per point
    assert res.values["sst"].shape == (1,)
    assert res.distance_km.shape == (1,)
    assert res.time_delta.shape == (1,)


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
    assert res.distance_km[1] > 25                    # p1: kept distance, beyond radius
    assert res.time_delta[2] > 86400                  # p2: kept time gap, beyond time_gap


def test_apply_on_2d_grid_ignores_time(grid_2d_ds):
    points = PointSet.from_arrays(lon=[30.0], lat=[2.0])

    res = NearestNeighbor().apply(grid_2d_ds, points)

    assert res.valid.tolist() == [True]
    npt.assert_allclose(res.values["v"], [230.0])
    assert np.isnan(res.time_delta[0])


# ---------- NearestNeighbor : k nearest nodes and time steps ----------
def test_k_nearest_gives_the_nearest_nodes_sorted(grid_3d_ds):
    # point (lon 21, lat 1.2) at the first time step
    #   node        distance    sst
    #   (20, 1)     ~113 km     120
    #   (20, 2)     ~142 km     220
    #   (20, 0)     ~173 km      20
    #   (30, 1)     ~1000 km    130   -> beyond radius
    points = PointSet.from_arrays(lon=[21.0], lat=[1.2],
                                  time=np.array(["2015-01-01"], dtype="datetime64[ns]"))

    res = NearestNeighbor(radius=500, k_nearest=4).apply(grid_3d_ds, points)

    assert res.values["sst"].shape == (1, 4, 1)
    npt.assert_allclose(res.values["sst"][0, :, 0], [120.0, 220.0, 20.0, np.nan])
    assert res.valid[0, :, 0].tolist() == [True, True, True, False]
    assert np.all(np.diff(res.distance_km[0]) > 0)      # sorted from the nearest, kept beyond radius


def test_n_frames_gives_the_nearest_time_steps(grid_3d_ds):
    # point on node (20, 1), 6 h after the first time step
    #   time step     gap     sst
    #   2015-01-01    6 h     120
    #   2015-01-02    18 h    121   -> beyond time_gap
    #   (none)                      -> 3 frames asked, only 2 in the grid: not found
    points = PointSet.from_arrays(lon=[20.0], lat=[1.0],
                                  time=np.array(["2015-01-01T06"], dtype="datetime64[ns]"))

    res = NearestNeighbor(time_gap=np.timedelta64(12, "h"), n_frames=3).apply(grid_3d_ds, points)

    assert res.values["sst"].shape == (1, 1, 3)
    npt.assert_allclose(res.values["sst"][0, 0], [120.0, np.nan, np.nan])
    npt.assert_allclose(res.time_delta[0], [6 * 3600, 18 * 3600, np.nan])
    assert res.valid[0, 0].tolist() == [True, False, False]


def test_k_nearest_and_n_frames_give_every_node_time_step_pair(grid_3d_ds):
    # point (lon 21, lat 1.2), 6 h after the first time step
    #                 2015-01-01 (6 h)   2015-01-02 (18 h)
    #   (20, 1)       120                121
    #   (20, 2)       220                221
    points = PointSet.from_arrays(lon=[21.0], lat=[1.2],
                                  time=np.array(["2015-01-01T06"], dtype="datetime64[ns]"))

    res = NearestNeighbor(radius=500, k_nearest=2, n_frames=2).apply(grid_3d_ds, points)

    assert res.values["sst"].shape == (1, 2, 2)
    npt.assert_allclose(res.values["sst"][0], [[120.0, 121.0],
                                               [220.0, 221.0]])
    assert res.valid[0].all()
    assert res.distance_km.shape == (1, 2)
    npt.assert_allclose(res.time_delta[0], [6 * 3600, 18 * 3600])


# ---------- Interpolator : IDW ----------
#
#            lon=0   lon=10
#   lat=10     30      40
#   lat=0      10      20

radius_wide = 5000  # km: wide enough that nothing is rejected


@pytest.fixture
def grid_square_ds():
    ds = make_grid([0.0, 10.0], [0.0, 10.0])
    ds["v"] = 2.0 * ds["lat"] + ds["lon"] + 10.0
    return ds


def _interp(grid, lon, lat, k=4, power=1.0):
    points = PointSet.from_arrays(lon=[lon], lat=[lat])
    return Interpolator(radius=radius_wide, k_nearest=k, power=power).apply(grid, points)


def test_interp_stays_in_the_range_of_the_four_nodes(grid_square_ds):
    res = _interp(grid_square_ds, lon=3.0, lat=4.0, k=4)
    assert 10.0 <= res.values["v"][0] <= 40.0


def test_interp_power_one_is_linear_halfway_between_two_nodes(grid_square_ds):
    # lon=5, lat=0 sits at equal distance from the nodes 10 and 20
    # -> with k=2 and power=1 the weights are equal, which is linear interpolation
    res = _interp(grid_square_ds, lon=5.0, lat=0.0, k=2)
    npt.assert_allclose(res.values["v"], [(10.0 + 20.0) / 2], rtol=1e-4)


def test_interp_k_limits_the_neighbours_even_inside_the_radius(grid_square_ds):
    two = _interp(grid_square_ds, lon=5.0, lat=0.0, k=2)

    # the 4 nodes are all inside radius, yet k=2 reads only the two closest
    npt.assert_allclose(two.values["v"], [(10.0 + 20.0) / 2], rtol=1e-4)



def test_interp_radius_limits_the_neighbours_below_the_requested_k(grid_square_ds):
    points = PointSet.from_arrays(lon=[5.0], lat=[0.0])
    # 800km
    res = Interpolator(radius=800, k_nearest=4, power=1.0).apply(
        grid_square_ds, points)

    # 4 asked, 2 are reachable
    npt.assert_allclose(res.values["v"], [(10.0 + 20.0) / 2], rtol=1e-4)
    assert res.valid[0]


def test_interp_more_frames_than_the_grid_does_not_fail(grid_3d_ds):
    # 3 frames asked, only 2 in the grid: the missing one is not found
    points = PointSet.from_arrays(lon=[20.0], lat=[1.0],
                                  time=np.array(["2015-01-01T06"], dtype="datetime64[ns]"))

    res = Interpolator(n_frames=3).apply(grid_3d_ds, points)

    assert res.valid.tolist() == [True]
    assert np.isfinite(res.values["sst"][0])
