# tests/conftest.py: fixtures shared across the test suite


import numpy as np
import pytest
import xarray as xr

from helpers import make_grid, daily_timestamps


# ---------- grids ----------

lat = [0.0, 1.0, 2.0]
lon = [10.0, 20.0, 30.0, 40.0]


@pytest.fixture
def grid_2d_ds():
    ds = make_grid(lat, lon)
    ds["v"] = 100.0 * ds["lat"] + ds["lon"]
    return ds


@pytest.fixture
def grid_3d_ds():
    ds = make_grid(lat, lon, time=daily_timestamps(2))
    ds["v"] = (100.0 * ds["lat"] + ds["lon"]
               + xr.DataArray(np.arange(ds.sizes["time"]), dims="time"))
    return ds