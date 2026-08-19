# geo.py : all geography transformations

# imports
import numpy as np 
from numpy.typing import ArrayLike, NDArray

# functions 
#_______
def lonlat_to_xyz(lon: ArrayLike, lat: ArrayLike) -> NDArray[np.float64]:
    """Convert lon/lat to xyz on the shpere"""
    R = 6371.0
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return np.column_stack([x, y, z])
