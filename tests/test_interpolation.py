# tests/test_interpolation.py

import numpy as np
import numpy.testing as npt
import xarray as xr

from floatmatcher.interpolation import Interpolation
from floatmatcher.gridset import GridSet
from floatmatcher.pointset import PointSet
from floatmatcher.method import Constraints
from floatmatcher.interpolation import pad_periodic_lon
 
 

# ───────────── test exactitude of interpolation ─────────────

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

# ───────────── test interp on inside/outside points ─────────────

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

# ───────────── test interpolation around a wrapping 360 longitudes axe ─────────────
 
def _grid(lon, nvars=3):
    lat = np.array([40.0, 30.0, 20.0])
    return xr.Dataset(
        {f"v{i}": (("lat", "lon"), np.random.rand(lat.size, lon.size)) for i in range(nvars)},
        coords={"lat": lat, "lon": lon},
    )
 
 
def test_pad_adds_one_column_on_global():
    ds = _grid(np.arange(0, 360, 0.25))
    padded = pad_periodic_lon(ds)
    assert padded.lon.size == ds.lon.size + 1
    assert float(padded.lon.max()) == 360.0
 
 
def test_pad_is_noop_on_regional():
    ds = _grid(np.arange(-100, 20, 0.25))
    padded = pad_periodic_lon(ds)
    assert padded.lon.size == ds.lon.size          
 
 
def test_pad_copies_every_variable():
    ds = _grid(np.arange(0, 360, 0.25), nvars=3)
    padded = pad_periodic_lon(ds)
    for v in ds.data_vars:
        # appended column (at 360) equals the first column (at 0), per variable
        assert np.array_equal(padded[v].isel(lon=-1).values, ds[v].isel(lon=0).values)
 
 
def test_pad_enables_interpolation_across_seam():
    ds = _grid(np.arange(0, 360, 0.25), nvars=1)
    p_lon = xr.DataArray([359.9], dims="p")
    p_lat = xr.DataArray([30.0], dims="p")
    # without padding: out of bounds -> NaN
    assert np.isnan(ds["v0"].interp(lon=p_lon, lat=p_lat).values[0])
    # with padding: interpolates between 359.75 and 360(==0) -> finite
    padded = pad_periodic_lon(ds)
    assert np.isfinite(padded["v0"].interp(lon=p_lon, lat=p_lat).values[0])
 
