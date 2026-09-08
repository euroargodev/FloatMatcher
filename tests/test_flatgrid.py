# tests/test_flatgrid.py
#
# FlatGrid flattens a grid into a node cloud and stays LAZY: read_values pulls
# only the retained (node[, time]) pairs, never the whole cube.

import numpy as np
import numpy.testing as npt

from floatmatcher.flatgrid import FlatGrid

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
    flat_grid_v = flat_grid.read_values([3, 4], [0, 1])["v"]
    expected_values = [40.0, 111.0]
    npt.assert_allclose(flat_grid_v, expected_values)



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
