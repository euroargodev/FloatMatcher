# tests/test_neighbors.py

import numpy as np
import numpy.testing as npt
import pytest

from floatmatcher.geo import lonlat_to_xyz
from floatmatcher.neighbors import spatial_nearest, temporal_nearest
from floatmatcher.pointset import PointSet

grid_lon = [0.0, 10.0, 0.0, 10.0]
grid_lat = [0.0, 0.0, 10.0, 10.0]
grid_xyz = lonlat_to_xyz(grid_lon, grid_lat)
grid_times = np.array(["2015-01-01", "2015-01-02", "2015-01-03",
                       "2015-01-04", "2015-01-05"], dtype="datetime64[ns]")


@pytest.fixture
def points_on_nodes():
    """Three points sitting exactly on grid nodes 2, 0 and 3."""
    return PointSet(
        lon=[0.0, 0.0, 10.0],
        lat=[10.0, 0.0, 10.0],
        time=np.array(["2015-01-03T03:00", "2015-01-01T00:00", "2015-01-04T23:00"],
                      dtype="datetime64[ns]"),
    )


def test_spatial_nearest_finds_the_node_under_each_point(points_on_nodes):
    dist, idx = spatial_nearest(grid_xyz, points_on_nodes)
    npt.assert_array_equal(idx, [2, 0, 3])
    npt.assert_allclose(dist, 0.0, atol=1e-9)


def test_spatial_nearest_distance_is_in_kilometres():
    points = PointSet(lon=[1.0], lat=[0.0],
                      time=np.array(["2015-01-01"], dtype="datetime64[ns]"))
    dist, idx = spatial_nearest(grid_xyz, points)
    expected = np.linalg.norm(points.xyz[0] - grid_xyz[0])
    npt.assert_array_equal(idx, [0])
    npt.assert_allclose(dist, [expected])


def test_spatial_nearest_k_returns_sorted_neighbours(points_on_nodes):
    dist, idx = spatial_nearest(grid_xyz, points_on_nodes, k=3)
    assert dist.shape == (3, 3)
    assert idx.shape == (3, 3)
    npt.assert_array_equal(idx[:, 0], [2, 0, 3])
    assert (np.diff(dist, axis=1) >= 0).all()


def test_temporal_nearest_delta_is_in_seconds(points_on_nodes):
    time_delta, idx = temporal_nearest(grid_times, points_on_nodes)
    npt.assert_array_equal(idx, [2, 0, 4])
    npt.assert_allclose(time_delta, [10800.0, 0.0, 3600.0])


def test_temporal_nearest_k_returns_sorted_neighbours(points_on_nodes):
    time_delta, idx = temporal_nearest(grid_times, points_on_nodes, k=2)
    assert idx.shape == (3, 2)
    npt.assert_array_equal(idx[:, 0], [2, 0, 4])
    npt.assert_allclose(time_delta[:, 0], [10800.0, 0.0, 3600.0])
    assert (np.diff(time_delta, axis=1) >= 0).all()
