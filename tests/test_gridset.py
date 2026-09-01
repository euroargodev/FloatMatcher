# tests/test_gridset.py
import pytest

from floatmatcher.gridset import GridSet


# ───────────── regime derivation ─────────────

def test_regime_3d(grid_3d_ds):
    """A grid with a time coord is 3D."""
    assert GridSet(grid_3d_ds).regime == "3D"

def test_regime_2d(grid_2d_ds):
    """A grid without a time coord is 2D."""
    assert GridSet(grid_2d_ds).regime == "2D"


# ───────────── validation ─────────────

def test_dataset_is_kept(grid_3d_ds):
    """GridSet wraps the dataset without transforming it."""
    grid = GridSet(grid_3d_ds)
    assert grid.dataset is grid_3d_ds          # same object, not a copy

def test_missing_lat_3d_raises(grid_3d_ds):
    """A grid without lat is rejected."""
    with pytest.raises(ValueError):
        GridSet(grid_3d_ds.drop_vars("lat"))

def test_missing_lat_2d_raises(grid_2d_ds):
    """A grid without lat is rejected."""
    with pytest.raises(ValueError):
        GridSet(grid_2d_ds.drop_vars("lat"))

def test_no_data_variable_raises(grid_2d_ds):
    """A grid with coords but no data variable is rejected."""
    with pytest.raises(ValueError):
        GridSet(grid_2d_ds.drop_vars("v"))   # removes the only variable


 
 
# ───────────── suffled and duplicated coordinates ─────────────
 
def test_shuffled_lon_is_accepted(grid_3d_ds):
    """ order does not matter: the KDTree works on a point cloud"""
    ds = grid_3d_ds.isel(lon=[2, 0, 1])            # [30,10,20] 
    assert GridSet(ds).regime == "3D"


def test_duplicate_lon_raises(grid_3d_ds):
    """two nodes at the same position would make the nearest lookup ambiguous"""
    ds = grid_3d_ds.isel(lon=[0, 1, 1])
    with pytest.raises(ValueError):
        GridSet(ds)
 
 
def test_decreasing_time_is_accepted(grid_3d_ds):
    """ decreasing time is accepted """
    ds = grid_3d_ds.isel(time=slice(None, None, -1))
    assert GridSet(ds).regime == "3D"
