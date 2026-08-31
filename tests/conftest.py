# tests/conftest.py: fixtures shared across the test suite


import numpy as np
import pytest
import xarray as xr
import pandas as pd
from floatmatcher.pointset import PointSet

from helpers import make_grid, daily_timestamps


# Reminder 
# 3D (grid_3d_ds): 
#        n0    n1    n2    n3     n4     n5     n6     n7     n8     n9    n10    n11
#  t=0  10.0  20.0  30.0  40.0  110.0  120.0  130.0  140.0  210.0  220.0  230.0  240.0
#  t=1  11.0  21.0  31.0  41.0  111.0  121.0  131.0  141.0  211.0  221.0  231.0  241.0

# 2D (grid_2d_ds) : 
#        n0    n1    n2    n3     n4     n5     n6     n7     n8     n9    n10    n11
#  t=0  10.0  20.0  30.0  40.0  110.0  120.0  130.0  140.0  210.0  220.0  230.0  240.0



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


@pytest.fixture
def points_with_origin():
    """A 3-point PointSet carrying provenance (as if extracted from a dataset)."""
    return PointSet(
        lon=[-45.0, -44.0, -43.0],
        lat=[32.0, 33.0, 34.0],
        time=np.array(["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]"),
        origin_dim="N_POINTS",
    )


# ---------- fake era5 ----------

@pytest.fixture
def write_era5(tmp_path):
    """Factory for ERA5-like NetCDF files."""
    def _write(name, times, lon, lat, values=None,
               variables=("u10", "t2m", "sst")):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        times = np.asarray(times, dtype="datetime64[ns]")
        shape = (times.size, lat.size, lon.size)

        if values is None:
            data = {}
            for v in variables:
                data[v] = np.random.rand(*shape).astype("float32")
        else:
            field = values(
                np.arange(lon.size)[None, None, :],
                np.arange(lat.size)[None, :, None],
                np.arange(times.size)[:, None, None],
            )
            field = (field * np.ones(shape)).astype("float32")
            data = {}
            for v in variables:
                data[v] = field

        data_vars = {}
        for v, d in data.items():
            data_vars[v] = (("valid_time", "latitude", "longitude"), d)

        ds = xr.Dataset(
            data_vars,
            coords={"valid_time": times, "latitude": lat, "longitude": lon},
        )
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)
        return str(path)

    return _write


@pytest.fixture
def fake_era5_file(write_era5):
    """Tiny ERA5-like NetCDF: raw names, 0-360 (a lon > 180), decreasing lat."""
    return write_era5(
        "fake_era5.nc",
        times=["2015-06-01T00", "2015-06-01T06"],
        lon=[9.0, 10.0, 11.0, 350.0],
        lat=[40.0, 30.0, 20.0],
    )



# ---------- interp lineaire ----------

@pytest.fixture
def argopy_like_ds():
    """xarray Dataset in the argopy N_PROF layout, WITH a real coordinate
    on the point dimension -> exercises label provenance."""
    n = 3
    return xr.Dataset(
        {
            "LONGITUDE": (("N_PROF",), np.array([-45.0, -44.0, -43.0])),
            "LATITUDE": (("N_PROF",), np.array([32.0, 33.0, 34.0])),
            "TIME": (("N_PROF",), np.array(
                ["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]")),
        },
        coords={"N_PROF": np.arange(n)},   # real coordinate on the point dim
    )
 
 
@pytest.fixture
def raw_ds_no_coord():
    """xarray Dataset on a non-standard dimension 'obs' WITHOUT a coordinate
    -> exercises dimension-name discovery AND positional provenance (arange)."""
    return xr.Dataset(
        {
            "LONGITUDE": (("obs",), np.array([-45.0, -44.0, -43.0])),
            "LATITUDE": (("obs",), np.array([32.0, 33.0, 34.0])),
            "TIME": (("obs",), np.array(
                ["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]")),
        },
        # no coords -> 'obs' is a bare dimension
    )

@pytest.fixture
def juld_ds():
    """xarray Dataset whose time variable is named JULD, not TIME
    -> exercises the TIME/JULD tolerance."""
    return xr.Dataset(
        {
            "LONGITUDE": (("obs",), np.array([-45.0, -44.0, -43.0])),
            "LATITUDE": (("obs",), np.array([32.0, 33.0, 34.0])),
            "JULD": (("obs",), np.array(
                ["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]")),
        },
    )
 
 
@pytest.fixture
def df_datetime_index():
    """DataFrame whose index is a DatetimeIndex (non-integer)"""
    idx = pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03"])
    idx.name = "profile_date"
    return pd.DataFrame(
        {
            "longitude": [-45.0, -44.0, -43.0],
            "latitude": [32.0, 33.0, 34.0],
            "date": idx,
        },
        index=idx,
    )