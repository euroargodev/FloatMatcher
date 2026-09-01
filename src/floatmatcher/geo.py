# geo.py : all geography transformations

# imports
import numpy as np 
from numpy.typing import ArrayLike, NDArray
from .constants import EARTH_RADIUS_KM

# functions 
#_______

def lonlat_to_xyz(lon: ArrayLike, lat: ArrayLike) -> NDArray[np.float64]:
    """Convert lon/lat to xyz on the shpere"""
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    x = EARTH_RADIUS_KM * np.cos(lat_rad) * np.cos(lon_rad)
    y = EARTH_RADIUS_KM * np.cos(lat_rad) * np.sin(lon_rad)
    z = EARTH_RADIUS_KM * np.sin(lat_rad)
    return np.column_stack([x, y, z])


# def convert_lon(lon: ArrayLike, lon_range: str) -> NDArray[np.float64]:
#     """Return longitude values re-labelled into the requested convention.
#     Both conventions share the 0 meridian (Greenwich) and turn eastward, no sorting.
#     Parameters
#     ----------
#     lon : array-like
#     lon_range : {"-180-180", "0-360"}
#     """
#     lon = np.asarray(lon, dtype=float)
#     if lon_range == "-180-180":
#         return ((lon + 180.0) % 360.0) - 180.0
#     if lon_range == "0-360":
#         return lon % 360.0
#     raise ValueError(
#         f"unknown lon_range: {lon_range!r} (expected '-180-180' or '0-360')"
#     )


# def detect_lon_range(lon: ArrayLike) -> str:
#     """Infer the longitude convention of a grid from its values.
#       - any value > 180  -> only "0-360" can produce that
#       - any value < 0    -> only "-180-180" can produce that
#       - all in [0, 180]  -> ambiguous (both conventions coincide there),
#                             default to "-180-180"
#     """
#     lon = np.asarray(lon, dtype=float)
#     if lon.max() > 180.0:
#         return "0-360"
#     if lon.min() < 0.0:
#         return "-180-180"
#     return "-180-180"
