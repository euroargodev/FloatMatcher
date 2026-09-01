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


