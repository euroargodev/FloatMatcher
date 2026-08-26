# tests/test_engine.py
import numpy as np
import pytest

from floatmatcher.nearest import NearestNeighbor
from floatmatcher.interpolation import Interpolation
from floatmatcher.method import MatchupMethod, Constraints
from floatmatcher.gridset import GridSet
from floatmatcher.pointset import PointSet

from helpers import set_timestamps


# both methods, driven through the SAME code path
ALL_METHODS = [NearestNeighbor(), Interpolation(method="linear")]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_method_honors_the_contract(method, standard_grid_3d):
    """Any method returns a MatchupResult aligned with the input points."""
    grid = GridSet(standard_grid_3d)
    lon = np.array([-46.5, -44.5])
    lat = np.array([33.5, 35.5])
    points = PointSet(lon, lat, set_timestamps(2, 2))

    result = method.match(grid, points, Constraints())

    # same output shape regardless of the method
    assert len(result.valid) == 2
    assert len(result.distance_km) == 2
    assert len(result.time_delta) == 2
    for var in grid.dataset.data_vars:
        assert len(result.values[var]) == 2


@pytest.mark.parametrize("method", ALL_METHODS)
def test_method_is_a_matchup_method(method):
    """Both concrete methods honor the abstract contract."""
    assert isinstance(method, MatchupMethod)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_out_of_grid_point_is_invalid(method, standard_grid_3d):
    """Both methods invalidate a point far outside the grid."""
    grid = GridSet(standard_grid_3d)
    lon = np.array([-48.0, 100.0])        # second point way outside
    lat = np.array([31.0, 34.0])
    points = PointSet(lon, lat, set_timestamps(2, 2))

    result = method.match(grid, points, Constraints())

    assert result.valid[0]
    assert not result.valid[1]
    assert np.isnan(result.values["v"][1])
    assert len(result.values["v"]) == 2   # masked, not dropped — for BOTH methods