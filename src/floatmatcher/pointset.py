# pointset.py: validated container for the input points to colocalize

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any
from .geo import lonlat_to_xyz
from .constants import TIME_UNIT


@dataclass
class PointSet:
    """Validated wrapper around the (lon, lat, time) arrays to colocalize.

    The wrapper carries validation and clear names, but heavy computation
    works directly on the underlying NumPy arrays (``points.lon``), never on
    the object itself inside loops.
    """

    lon: NDArray[np.float64]
    lat: NDArray[np.float64]
    time: NDArray[np.datetime64]
    origin_index: NDArray[Any] | None = None   # source index, if any
    origin_dim: str | None = None                   # source dimension name, if known
    _xyz: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """validation for lenghts and dtype of lon/lat/time"""
        self.lon = np.asarray(self.lon, dtype=float)
        self.lat = np.asarray(self.lat, dtype=float)
        self.time = np.asarray(self.time, dtype=f"datetime64[{TIME_UNIT}]")
        if not (len(self.lat) == len(self.time) == len(self.lon)):
            raise ValueError("lon, lat, time must have the same length")
        if self.origin_index is not None:
            self.origin_index = np.asarray(self.origin_index)

    @property
    def xyz(self) -> NDArray[np.float64]:
        """Cartesian 3D coordinates on the sphere, computed once and cached."""
        if self._xyz is None:
            self._xyz = lonlat_to_xyz(self.lon, self.lat)
        return self._xyz
    