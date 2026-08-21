# reference.py: the reference point cloud (flattened grid), internal to nearest

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .geo import lonlat_to_xyz


@dataclass
class ReferenceSet:
    """Flattened grid as a point cloud, for KDTree lookup."""

    lon: NDArray[np.float64]
    lat: NDArray[np.float64]
    time: NDArray[np.datetime64] | None      
    values: dict[str, NDArray[np.float64]]
    _xyz: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    @property
    def xyz(self) -> NDArray[np.float64]:
        """Cartesian 3D coordinates of the nodes, computed once and cached."""
        if self._xyz is None:
            self._xyz = lonlat_to_xyz(self.lon, self.lat)
        return self._xyz


def grid_to_reference(ds: xr.Dataset) -> ReferenceSet:
    """Flatten a grid (2D or 3D) into a ReferenceSet."""
    stacked = ds.stack(node=("lat", "lon"))
    lon = stacked["lon"].values
    lat = stacked["lat"].values

    if "time" in ds.coords:
        time = ds["time"].values
        # after stack, a variable is (time, node); we want (node, time)
        values = {var: stacked[var].transpose("node", "time").values
                  for var in ds.data_vars}
    else:
        time = None
        values = {var: stacked[var].values for var in ds.data_vars}

    return ReferenceSet(lon=lon, lat=lat, time=time, values=values)