# tests/test_gridset.py
import numpy as np
import pytest
import xarray as xr

from floatmatcher.gridset import GridSet


# --- regime derivation ---
def test_regime_3d(grid_3d_ds):
    """A grid with a time coord is 3D."""
    assert GridSet(grid_3d_ds).regime == "3D"

def test_regime_2d(grid_2d_ds):
    """A grid without a time coord is 2D."""
    assert GridSet(grid_2d_ds).regime == "2D"

# --- the dataset is kept intact ---
def test_dataset_is_kept(grid_3d_ds):
    """GridSet wraps the dataset without transforming it."""
    grid = GridSet(grid_3d_ds)
    assert grid.dataset is grid_3d_ds          # same object, not a copy


# --- validation ---
def test_missing_lat_raises(grid_3d_ds):
    """A grid without lat is rejected."""
    with pytest.raises(ValueError):
        GridSet(grid_3d_ds.drop_vars("lat"))

def test_missing_lon_raises(grid_3d_ds):
    """A grid without lon is rejected."""
    with pytest.raises(ValueError):
        GridSet(grid_3d_ds.drop_vars("lon"))

def test_no_data_variable_raises(grid_2d_ds):
    """A grid with coords but no data variable is rejected."""
    with pytest.raises(ValueError):
        GridSet(grid_2d_ds.drop_vars("2DdummyVar")) # removes the only variable