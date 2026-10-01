
import numpy as np 
import xarray as xr

def random_positions(n_points, lon_min=-180, lon_max=180, lat_min=-70, lat_max=90):

    rng = np.random.default_rng(0)

    lon = rng.uniform(lon_min, lon_max, n_points)
    lat = rng.uniform(lat_min, lat_max, n_points)
    day_offsets = rng.integers(0, 10, size=n_points).astype("timedelta64[D]")
    hour_offsets = rng.integers(0, 24, size=n_points).astype("timedelta64[h]")
    time = (np.datetime64("2018-01-01") + day_offsets + hour_offsets).astype("datetime64[ns]")

    return lon, lat, time


def random_dataset():
    ds_profiles = xr.Dataset(
        {
            "LONGITUDE": (("N_PROF",), np.array([-40.0, -30.0, -20.0])),
            "LATITUDE": (("N_PROF",), np.array([35.0, 36.0, 37.0])),
            "TIME": (("N_PROF",), np.array(["2018-01-03", "2018-01-04", "2018-01-05"],
                                        dtype="datetime64[ns]")),
        },
        coords={"N_PROF": np.array([10, 11, 12])},
    )
    return ds_profiles