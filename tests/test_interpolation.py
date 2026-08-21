# tests/test_interpolation.py

import numpy as np
import numpy.testing as npt

from floatmatcher.interpolation import Interpolation
from floatmatcher.gridset import GridSet
from floatmatcher.pointset import PointSet
from floatmatcher.method import Constraints


def _linear(lon, lat):
    return 2.0 * lon + 3.0 * lat


def test_linear_field_is_interpolated_exactly(grid_3d_equalalltimes):
    """Linear interpolation of a linear field is exact at any point."""
    grid = GridSet(grid_3d_equalalltimes)
    # points at half-integer positions: the hardest case (mid-cell)
    lon = np.array([-45.5, -47.2, -49.5])  
    lat = np.array([32.5, 35.8, 37.5])     
    time = np.array(["2015-01-01", "2015-01-01", "2015-01-02"], dtype="datetime64[ns]")
    points = PointSet(lon, lat, time)

    result = Interpolation(method="linear").match(grid, points, Constraints())

    expected = _linear(lon, lat)
    npt.assert_allclose(result.values["v"], expected, atol=1e-9)
    assert result.valid.all()


def test_point_outside_grid_is_invalid(grid_3d_equalalltimes):
    """A point outside the grid footprint comes out NaN and invalid."""
    grid = GridSet(grid_3d_equalalltimes)
    lon = np.array([-45.0, 0.0])          
    lat = np.array([35.0, 35.0])
    time = np.array(["2015-01-01", "2015-01-01"], dtype="datetime64[ns]")
    points = PointSet(lon, lat, time)

    result = Interpolation().match(grid, points, Constraints())

    assert result.valid[0]                     # inside → valid
    assert not result.valid[1]                 # outside → invalid
    assert np.isnan(result.values["v"][1])     # and its value is NaN