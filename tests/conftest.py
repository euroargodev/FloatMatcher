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
def points_object():
    """A 3-point PointSet carrying provenance (as if extracted from a dataset)."""
    return PointSet(
        lon=[-45.0, -44.0, -43.0],
        lat=[32.0, 33.0, 34.0],
        time=np.array(["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]"),
        origin_dim="N_POINTS",
    )


# ---------- fake era5 ----------

@pytest.fixture
def era5_files(tmp_path):
    """Five daily ERA5-like files (2015/06/01 to 2015/06/05).
    first timestep : 
              lon=9     lon=10   lon=11    lon=350
    lat=40    1.03      11.03     21.03     31.03
    lat=30    101.03    111.03    121.03    131.03
    lat=20    201.03    211.03    221.03    231.03
    """
    lat = [40.0, 30.0, 20.0]
    lon = [9.0, 10.0, 11.0, 350.0]
    sst = np.array([[  0.,  10.,  20.,  30.],
                    [100., 110., 120., 130.],
                    [200., 210., 220., 230.]])

    paths = []
    for day in range(1, 6):
        times = np.array([f"2015-06-{day:02d}T03", f"2015-06-{day:02d}T23"],
                         dtype="datetime64[ns]")
        shape_lst = list(np.shape(sst))
        shape_lst.append(len(times))
        shape = tuple(shape_lst)
        field = np.zeros(shape)

        for t in range(len(times)) :
            h = times[t].astype("datetime64[h]").astype(int) % 24 /100
            field[:,:,t] = sst+day+h
        dims = ("latitude", "longitude", "valid_time")
        ds = xr.Dataset(
            {"sst": (dims, field), "t2m": (dims, field + 1000.0)},
            coords={"latitude": lat, "longitude": lon, "valid_time": times},
        )
        path = tmp_path / f"era5_201506{day:02d}.nc"
        ds.to_netcdf(path)
        paths.append(str(path))

    return paths


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