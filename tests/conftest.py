# tests/conftest.py: fixtures shared across the test suite
import numpy as np
import pytest
import xarray as xr
import pandas as pd
from floatmatcher.pointset import PointSet


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


@pytest.fixture
def points_with_origin():
    """A 3-point PointSet carrying provenance (as if extracted from a dataset)."""
    return PointSet(
        lon=[-45.0, -44.0, -43.0],
        lat=[32.0, 33.0, 34.0],
        time=np.array(["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]"),
        origin_index=np.array([0, 1, 2]),
        origin_dim="N_POINTS",
    )

@pytest.fixture
def grid_0_360():
    """Grid whose longitudes exceed 180 -> native 0-360 convention.
    Cannot be derived from grid_3d_ds (whose lon stays in [0,180])."""
    lat = np.array([0.0, 1.0])
    lon = np.array([200.0, 201.0, 202.0])
    data = np.zeros((lat.size, lon.size))
    return xr.Dataset({"v": (("lat", "lon"), data)},
                      coords={"lat": lat, "lon": lon})


@pytest.fixture
def fake_era5_file(tmp_path):
    """Tiny ERA5-like NetCDF: raw names, 0-360 (a lon > 180), decreasing lat."""
    lat = np.array([40.0, 30.0, 20.0])
    lon = np.array([9.0, 10.0, 11.0, 350.0])      # 350 > 180 -> 0-360
    vt = np.array(["2015-06-01T00", "2015-06-01T06"], dtype="datetime64[ns]")
    shape = (vt.size, lat.size, lon.size)
    ds = xr.Dataset(
        {v: (("valid_time", "latitude", "longitude"),
             np.random.rand(*shape).astype("float32"))
         for v in ["u10", "t2m", "sst"]},
        coords={"valid_time": vt, "latitude": lat, "longitude": lon},
    )
    p = tmp_path / "fake_era5.nc"
    ds.to_netcdf(p)
    return str(p)


@pytest.fixture
def points_in_grid():
    return PointSet(
        lon=[9.5, 10.5],
        lat=[35.0, 25.0],
        time=np.array(["2015-06-01T03", "2015-06-01T03"], dtype="datetime64[ns]"),
    )


# ---------- interp lineaire ----------
def _linear(lon, lat):
    """Known analytic field: value = 2*lon + 3*lat."""
    return 2.0 * lon + 3.0 * lat

@pytest.fixture
def grid_3d_equalalltimes():
    """3D grid whose variable is an exact linear function of position.

    The value does not depend on time here, which is fine: it lets us predict
    the interpolated result from position alone.
    """
    lat = np.arange(30.0, 40.0)          # 30..39
    lon = np.arange(-50.0, -40.0)        # -50..-41
    time = np.array([np.datetime64("2015-01-01"), np.datetime64("2015-01-02"),
                     np.datetime64("2015-01-03")], dtype="datetime64[ns]")
    lon2d, lat2d = np.meshgrid(lon, lat)          # (lat, lon)
    field2d = _linear(lon2d, lat2d)               # (lat, lon)
    data = np.broadcast_to(field2d, (time.size, lat.size, lon.size))
    return xr.Dataset(
        {"v": (("time", "lat", "lon"), data.copy())},
        coords={"time": time, "lat": lat, "lon": lon},
    )


# ---------- standard grids ----------
@pytest.fixture
def standard_grid_2d():
    """2D grid where each node's value encodes its position: 1000*lat + lon."""
    lat = np.array([30.0, 31.0, 32.0])
    lon = np.array([-50.0, -49.0, -48.0, -47.0])
    lon2d, lat2d = np.meshgrid(lon, lat)
    field = 1000.0 * lat2d + lon2d
    return xr.Dataset({"v": (("lat", "lon"), field)},
                      coords={"lat": lat, "lon": lon})


@pytest.fixture
def standard_grid_3d():
    """3D grid where value encodes position and time index: 1000*lat + lon + 100000*t."""
    lat = np.array([30.0, 31.0, 32.0])
    lon = np.array([-50.0, -49.0, -48.0, -47.0])
    time = np.array(["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]")
    field = np.empty((time.size, lat.size, lon.size))
    lon2d, lat2d = np.meshgrid(lon, lat)
    for t in range(time.size):
        field[t] = 1000.0 * lat2d + lon2d + 100000.0 * t
    return xr.Dataset({"v": (("time", "lat", "lon"), field)},
                      coords={"time": time, "lat": lat, "lon": lon})



# ---------- fixtures for profile_loader ----------

@pytest.fixture
def argopy_like_ds():
    """xarray Dataset in the argopy N_POINTS layout, WITH a real coordinate
    on the point dimension -> exercises label provenance."""
    n = 3
    return xr.Dataset(
        {
            "LONGITUDE": (("N_POINTS",), np.array([-45.0, -44.0, -43.0])),
            "LATITUDE": (("N_POINTS",), np.array([32.0, 33.0, 34.0])),
            "TIME": (("N_POINTS",), np.array(
                ["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]")),
        },
        coords={"N_POINTS": np.arange(n)},   # real coordinate on the point dim
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
    """DataFrame whose index is a DatetimeIndex (non-integer)
    -> exercises the 'soft' policy: origin_index carried as-is, not cast to int."""
    idx = pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03"])
    idx.name = "profile_date"
    return pd.DataFrame(
        {
            "longitude": [-45.0, -44.0, -43.0],
            "latitude": [32.0, 33.0, 34.0],
            "date": pd.to_datetime(["2015-06-01", "2015-06-02", "2015-06-03"]),
        },
        index=idx,
    )
 
