# tests/helpers.py: helpers shared across the test folder
import numpy as np
import xarray as xr


def set_timestamps(*days):
    """Timestamps for the given 2015-01-xx days."""
    stamps = []
    for d in days:
        stamps.append(f"2015-01-{d:02d}")
    return np.array(stamps, dtype="datetime64[ns]")


def daily_timestamps(n):
    """n consecutive daily timestamps starting 2015-01-01."""
    return set_timestamps(*range(1, n + 1))


def linear_field(lon, lat):
    """Analytic field: 2*lon + 3*lat."""
    return 2.0 * lon + 3.0 * lat


def pos_field(lon2d, lat2d, t, lat_coef, time_coef):
    """Encode node position in its value: lat_coef*lat + lon + time_coef*t."""
    return lat_coef * lat2d + lon2d + time_coef * t


def _build_field(lon2d, lat2d, time, fill, fill_args):
    if time is None:
        if fill is None:
            return np.zeros(lon2d.shape)
        return np.asarray(fill(lon2d, lat2d, 0, *fill_args), dtype=float)
    slices = []
    for t in range(time.size):
        if fill is None:
            slices.append(np.zeros(lon2d.shape))
        else:
            slices.append(np.asarray(fill(lon2d, lat2d, t, *fill_args), dtype=float))
    return np.stack(slices)
 
 
def make_grid(lat, lon, time=None, variables=("v",), fill=None, fill_args=()):
    """Build a GridSet-ready Dataset. 2D when time is None, else 3D.
 
    fill(lon2d, lat2d, t, *fill_args) returns the (lat, lon) field for time
    index t; None fills with zeros.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if isinstance(variables, str):
        variables = (variables,)
    lon2d, lat2d = np.meshgrid(lon, lat)
 
    if time is None:
        dims = ("lat", "lon")
        coords = {"lat": lat, "lon": lon}
    else:
        time = np.asarray(time, dtype="datetime64[ns]")
        dims = ("time", "lat", "lon")
        coords = {"time": time, "lat": lat, "lon": lon}
 
    data_vars = {}
    for name in variables:
        data_vars[name] = (dims, _build_field(lon2d, lat2d, time, fill, fill_args))
 
    return xr.Dataset(data_vars, coords=coords)
 
