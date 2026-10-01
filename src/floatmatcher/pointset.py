# pointset.py: validated container for the input points to colocalize

from dataclasses import dataclass, field
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from .geo import lonlat_to_xyz
from .constants import TIME_UNIT


class PointSet:
    """Validated wrapper around the (lon, lat, time) arrays to colocalize.

    The wrapper carries validation and clear names, but heavy computation
    works directly on the underlying NumPy arrays (``points.lon``), never on
    the object itself inside loops.

    Dataset travels along origin_ds
    """
    def __init__(self, 
                lon: NDArray[np.float64],
                lat: NDArray[np.float64],
                time: NDArray[np.datetime64] | None = None
            ):
        self.lon = lon
        self.lat = lat 
        self.time = time 
        self.check_lengths()

    # TODO: introduce PointSetOrigin class (ds and df)  
    origin_dim: str | None = None       # source dimension name
    origin_ds: xr.Dataset | None = None  # source dataset
    _xyz: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def check_lengths(self,):
        # TODO : write test
        if self.time is not None: 
            lengths = [len(self.lon), len(self.lat), len(self.time)]
        else:
            lengths = [len(self.lon), len(self.lat)]

        if len(np.unique(lengths)) != 1:
            raise ValueError("lon, lat, time must have the same length")

    # TODO : 
    def __repr__(self,) -> str:
        if self.time is not None:
            return f"lon:{len(self.lon)} \nlat:{len(self.lat)} \ntime:{len(self.time)}"
        else:
            return f"lon:{len(self.lon)} \nlat:{len(self.lat)}"
                
    @property
    def xyz(self) -> NDArray[np.float64]:
        """Cartesian 3D coordinates on the sphere, computed once and cached."""
        if self._xyz is None:
            self._xyz = lonlat_to_xyz(self.lon, self.lat)
        return self._xyz


    # --- 1. Raw arrays: simplest case ---
    def from_arrays(self, 
                    lon: NDArray[np.float64], 
                    lat: NDArray[np.float64], 
                    time:NDArray[np.datetime64] | None = None ):
        # Nothing to extract, no origin_* (bare arrays).
        self.lon = lon
        self.lat = lat 
        if time:
            self.time = time
        self.check_lengths()
