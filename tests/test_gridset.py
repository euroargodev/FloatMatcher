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

# --- the dataset is kept intact ---
def test_dataset_is_kept(grid_3d_ds):
    """GridSet wraps the dataset without transforming it."""
    grid = GridSet(grid_3d_ds)
    assert grid.dataset is grid_3d_ds          # same object, not a copy



# ───────────── validation ─────────────

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


 
 
# ───────────── monotonicity of lat/lon ─────────────
 
def test_shuffled_lon_is_accepted(grid_3d_ds):
    # order does not matter: the KDTree works on a point cloud
    ds = grid_3d_ds.isel(lon=[2, 0, 1])            # [12,10,11] -> not monotonic
    assert GridSet(ds).regime == "3D"


def test_duplicate_lon_raises(grid_3d_ds):
    # two nodes at the same position would make the nearest lookup ambiguous
    ds = grid_3d_ds.isel(lon=[0, 1, 1])
    with pytest.raises(ValueError):
        GridSet(ds)
 
 
def test_decreasing_lat_is_accepted(grid_3d_ds):
    # ERA5-like latitude running north -> south is valid
    ds = grid_3d_ds.isel(lat=slice(None, None, -1))
    assert GridSet(ds).regime == "3D"
 
 
# ───────────── time monotonicity (3D only) ─────────────
 
def test_decreasing_time_is_accepted(grid_3d_ds):
    ds = grid_3d_ds.isel(time=slice(None, None, -1))
    assert GridSet(ds).regime == "3D"


def test_duplicate_time_raises(grid_3d_ds):
    # overlapping files would silently return either value
    ds = grid_3d_ds.isel(time=[0, 0])
    with pytest.raises(ValueError):
        GridSet(ds)
 
 
def test_time_check_skipped_in_2d(grid_2d_ds):
    # a 2D grid has no time -> the time check must not run and must not fail
    assert GridSet(grid_2d_ds).regime == "2D"
 
