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




def make_grid(lat, lon, time=None, variables=("v",)):
    """Build a GridSet-ready Dataset filled with zeros.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if isinstance(variables, str):
        variables = (variables,)

    if time is None:
        dims = ("lat", "lon")
        coords = {"lat": lat, "lon": lon}
        shape = (lat.size, lon.size)
    else:
        time = np.asarray(time, dtype="datetime64[ns]")
        dims = ("time", "lat", "lon")
        coords = {"time": time, "lat": lat, "lon": lon}
        shape = (time.size, lat.size, lon.size)

    data_vars = {}
    for name in variables:
        data_vars[name] = (dims, np.zeros(shape))

    return xr.Dataset(data_vars, coords=coords)
