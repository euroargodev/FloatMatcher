# tests/test_flatgrid.py
#
# FlatGrid flattens a grid into a node cloud and stays LAZY: read_values pulls
# only the retained (node[, time]) pairs, never the whole cube.

import numpy as np
import numpy.testing as npt
import pytest

from floatmatcher.flatgrid import FlatGrid
from floatmatcher.pointset import PointSet
from helpers import make_grid

# Reminder 
# 3D (grid_3d_ds): 
#        n0    n1    n2    n3     n4     n5     n6     n7     n8     n9    n10    n11
#  t=0  10.0  20.0  30.0  40.0  110.0  120.0  130.0  140.0  210.0  220.0  230.0  240.0
#  t=1  11.0  21.0  31.0  41.0  111.0  121.0  131.0  141.0  211.0  221.0  231.0  241.0

# 2D (grid_2d_ds) : 
#        n0    n1    n2    n3     n4     n5     n6     n7     n8     n9    n10    n11
#  t=0  10.0  20.0  30.0  40.0  110.0  120.0  130.0  140.0  210.0  220.0  230.0  240.0


def test_flatten_preserves_alignment(grid_2d_ds):
    """Each flattened node keeps the value of its own coordinates"""
    flat_grid = FlatGrid.from_grid(grid_2d_ds)

    assert flat_grid.lon.size == 3 * 4                     # lat * lon

    values = flat_grid.read_values(np.arange(12))["v"]     # 2D: no time index
    npt.assert_allclose(values, [10.0, 20.0, 30.0, 40.0,
                                 110.0, 120.0, 130.0, 140.0,
                                 210.0, 220.0, 230.0, 240.0])


def test_flatgrid_is_2d_has_no_time(grid_2d_ds):
    """A 2D grid produces a FlatGrid with time=None."""
    assert FlatGrid.from_grid(grid_2d_ds).time is None


def test_flatgrid_xyz_shape(grid_2d_ds):
    """xyz exposes one (x, y, z) row per node."""
    assert FlatGrid.from_grid(grid_2d_ds).xyz.shape == (12, 3)


def test_retreive_right_indexes(grid_3d_ds):
    """Indices are paired element-wise, not crossed: node[i] with time[i]"""
    flat_grid = FlatGrid.from_grid(grid_3d_ds)

    # node 3 read at t=0 -> 40 ; node 4 read at t=1 -> 111
    flat_grid_sst = flat_grid.read_values([3, 4], [0, 1])["sst"]
    expected_values = [40.0, 111.0]
    npt.assert_allclose(flat_grid_sst, expected_values)



def test_read_values_returns_every_variable(grid_2d_ds):
    """read_values loops over data_vars"""
    grid_2d_ds["sst"] = grid_2d_ds["v"]
    grid_2d_ds["t2m"] = grid_2d_ds["v"] + 1000.0

    flat_grid = FlatGrid.from_grid(grid_2d_ds)
    grid_values = flat_grid.read_values([0, 4]) # read node 0 and 4 on 2d grid --> dict

    assert set(grid_values) == {"v", "sst", "t2m"}
    npt.assert_allclose(grid_values["sst"], [10.0, 110.0])
    npt.assert_allclose(grid_values["t2m"], [1010.0, 1110.0])


def test_xyz_is_cached(grid_2d_ds):
    """xyz is computed once: the same array object comes back.
    test if @property works well"""
    flat_grid = FlatGrid.from_grid(grid_2d_ds)
    assert flat_grid.xyz is flat_grid.xyz



# ---------- nearest lookups ----------
# 2x2 grid, nodes (lat, lon): 0 = (0, 0), 1 = (0, 10), 2 = (10, 0), 3 = (10, 10)

GRID_TIMES = np.array(["2015-01-01", "2015-01-02", "2015-01-03",
                       "2015-01-04", "2015-01-05"], dtype="datetime64[ns]")


@pytest.fixture
def flat_2d():
    return FlatGrid.from_grid(make_grid(lat=[0.0, 10.0], lon=[0.0, 10.0]))


@pytest.fixture
def flat_3d():
    return FlatGrid.from_grid(make_grid(lat=[0.0, 10.0], lon=[0.0, 10.0], time=GRID_TIMES))


@pytest.fixture
def points_on_nodes():
    """Three points sitting exactly on grid nodes 2, 0 and 3."""
    return PointSet(
        lon=[0.0, 0.0, 10.0],
        lat=[10.0, 0.0, 10.0],
        time=np.array(["2015-01-03T03:00", "2015-01-01T00:00", "2015-01-04T23:00"],
                      dtype="datetime64[ns]"),
    )


def test_spatial_nearest_finds_the_node_under_each_point(flat_2d, points_on_nodes):
    dist, idx = flat_2d.spatial_nearest(points_on_nodes)
    npt.assert_array_equal(idx, [2, 0, 3])
    npt.assert_allclose(dist, 0.0, atol=1e-9)


def test_spatial_nearest_distance_is_in_kilometres(flat_2d):
    points = PointSet(lon=[1.0], lat=[0.0],
                      time=np.array(["2015-01-01"], dtype="datetime64[ns]"))
    dist, idx = flat_2d.spatial_nearest(points)
    expected = np.linalg.norm(points.xyz[0] - flat_2d.xyz[0])
    npt.assert_array_equal(idx, [0])
    npt.assert_allclose(dist, [expected])


def test_spatial_nearest_k_returns_sorted_neighbours(flat_2d, points_on_nodes):
    dist, idx = flat_2d.spatial_nearest(points_on_nodes, k=3)
    assert dist.shape == (3, 3)
    assert idx.shape == (3, 3)
    npt.assert_array_equal(idx[:, 0], [2, 0, 3])
    assert (np.diff(dist, axis=1) >= 0).all()


def test_temporal_nearest_delta_is_in_seconds(flat_3d, points_on_nodes):
    time_delta, idx = flat_3d.temporal_nearest(points_on_nodes)
    npt.assert_array_equal(idx, [2, 0, 4])
    npt.assert_allclose(time_delta, [10800.0, 0.0, 3600.0])


def test_temporal_nearest_k_returns_sorted_neighbours(flat_3d, points_on_nodes):
    time_delta, idx = flat_3d.temporal_nearest(points_on_nodes, k=2)
    assert idx.shape == (3, 2)
    npt.assert_array_equal(idx[:, 0], [2, 0, 4])
    npt.assert_allclose(time_delta[:, 0], [10800.0, 0.0, 3600.0])
    assert (np.diff(time_delta, axis=1) >= 0).all()
