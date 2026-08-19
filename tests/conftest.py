# tests/conftest.py: fixtures shared across the test suite
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def grid_3d_ds():
    """Minimal well-formed 3D grid dataset (time/lat/lon), one variable."""
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([10.0, 11.0, 12.0])
    time = np.array([np.datetime64("2015-01-01"), np.datetime64("2015-01-02")])
    data = np.zeros((time.size, lat.size, lon.size))
    return xr.Dataset(
        {"t2m": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


@pytest.fixture
def grid_2d_ds():
    """Minimal well-formed 2D grid dataset (lat/lon, no time), one variable."""
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([10.0, 11.0, 12.0])
    data = np.zeros((lat.size, lon.size))
    return xr.Dataset(
        {"2DdummyVar": (("lat", "lon"), data)},
        coords={"lat": lat, "lon": lon},
    )