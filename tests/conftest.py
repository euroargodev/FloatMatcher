# tests/conftest.py: fixtures shared across the test suite
import numpy as np
import pytest
import xarray as xr
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


# tests/conftest.py  (à ajouter)

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