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
    _stacked: xr.Dataset                      
    _xyz: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    @property
    def xyz(self) -> NDArray[np.float64]:
        """Cartesian 3D coordinates of the nodes, computed once and cached."""
        if self._xyz is None:
            self._xyz = lonlat_to_xyz(self.lon, self.lat)
        return self._xyz

    def read_values(self, node_idx, time_idx=None) -> dict[str, NDArray[np.float64]]:
        """Read variable values ONLY at the retained (node[, time]) indices.
 
        Vectorized isel over a shared 'pts' dimension: on a lazy dataset this
        materializes just the selected points, not the full cube. `time_idx` is
        None for 2D grids, an array aligned with `node_idx` for 3D.
        """
        node = xr.DataArray(np.asarray(node_idx), dims="pts")
        out = {}
        for var in self._stacked.data_vars:
            da_var = self._stacked[var]
            if time_idx is None:
                sel = da_var.isel(node=node)
            else:
                sel = da_var.isel(node=node, time=xr.DataArray(np.asarray(time_idx), dims="pts"))
            out[var] = sel.values
        return out



def grid_to_reference(ds: xr.Dataset) -> ReferenceSet:
    """Flatten a grid (2D or 3D) into a ReferenceSet.
    Only coordinates are materialized here 
    Variable values stay lazy in the stacked dataset and are read later at the retained nodes.
    """
    stacked = ds.stack(node=("lat", "lon"))
    lon = stacked["lon"].values
    lat = stacked["lat"].values
    time = ds["time"].values if "time" in ds.coords else None
    return ReferenceSet(lon=lon, lat=lat, time=time, _stacked=stacked)
