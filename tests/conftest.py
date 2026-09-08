# tests/conftest.py: fixtures shared across the test suite


import numpy as np
import pytest
import xarray as xr

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
