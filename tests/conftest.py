# tests/conftest.py: fixtures shared across the test suite


import numpy as np
import pytest
import xarray as xr
import pandas as pd
from floatmatcher.pointset import PointSet

from helpers import make_grid, linear_field, pos_field, daily_timestamps
 

# ---------- grids ----------

@pytest.fixture
def grid_3d_ds():
    return make_grid([0.0, 1.0, 2.0], [10.0, 11.0, 12.0],
                     time=daily_timestamps(2), variables="t2m")

@pytest.fixture
def grid_2d_ds():
    return make_grid([0.0, 1.0, 2.0], [10.0, 11.0, 12.0], variables="2DdummyVar")

@pytest.fixture
def grid_0_360():
    return make_grid([0.0, 1.0], [200.0, 201.0, 202.0])

def _equal_all_times(lon2d, lat2d, t):
    return linear_field(lon2d, lat2d)




# ---------- points ----------

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
def points_in_grid():
    return PointSet(
        lon=[9.5, 10.5],
        lat=[35.0, 25.0],
        time=np.array(["2015-06-01T03", "2015-06-01T03"], dtype="datetime64[ns]"),
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
def grid_3d_equalalltimes():
    return make_grid(np.arange(30.0, 40.0), np.arange(-50.0, -40.0),
                     time=daily_timestamps(3), fill=_equal_all_times)



# ---------- standard grids ----------

@pytest.fixture
def standard_grid_2d():
    return make_grid([30.0, 31.0, 32.0], [-50.0, -49.0, -48.0, -47.0],
                     fill=pos_field, fill_args=(1000.0, 0.0))
 
 
@pytest.fixture
def standard_grid_3d():
    return make_grid([30.0, 31.0, 32.0], [-50.0, -49.0, -48.0, -47.0],
                     time=daily_timestamps(3),
                     fill=pos_field, fill_args=(1000.0, 100000.0))
 
 



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