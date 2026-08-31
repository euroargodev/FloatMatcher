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



# ───────────── lon_range derivation ─────────────
 
def test_lon_range_ambiguous_defaults_180(grid_3d_ds):
    # lon [10,11,12] all in [0,180] -> ambiguous -> default -180-180
    assert GridSet(grid_3d_ds).lon_range == "-180-180"
 
def test_lon_range_0_360(grid_0_360):
    assert GridSet(grid_0_360).lon_range == "0-360"
 
def test_lon_range_negative_is_180(grid_3d_ds):
    # shift [10,11,12] -> [-50,-49,-48] to get a negative-lon grid
    ds = grid_3d_ds.assign_coords(lon=grid_3d_ds.lon - 60.0)
    assert GridSet(ds).lon_range == "-180-180"

def test_declared_none_skips_check(grid_0_360):
    # no declared convention -> GridSet just derives from data, no check
    assert GridSet(grid_0_360).lon_range == "0-360"
 
def test_declared_matches_data(grid_0_360):
    # data is 0-360 and product declares 0-360 -> ok
    grid = GridSet(grid_0_360, declared_lon_range="0-360")
    assert grid.lon_range == "0-360"
 
def test_declared_mismatch_raises(grid_0_360):
    # data is 0-360 but product declares -180-180 -> anomaly, must raise
    with pytest.raises(ValueError):
        GridSet(grid_0_360, declared_lon_range="-180-180")
 
def test_declared_matches_negative(grid_3d_ds):
    # shift lon into negatives, declare -180-180 -> ok
    ds = grid_3d_ds.assign_coords(lon=grid_3d_ds.lon - 60.0)   # -> [-50,-49,-48]
    grid = GridSet(ds, declared_lon_range="-180-180")
    assert grid.lon_range == "-180-180"

 
 
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
 
