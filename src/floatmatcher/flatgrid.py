# reference.py: the reference point cloud (flattened grid), internal to nearest

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from .geo import lonlat_to_xyz


@dataclass
class FlatGrid:
    """Flattened grid as a point cloud, for KDTree lookup."""

    lon: NDArray[np.float64]
    lat: NDArray[np.float64]
    time: NDArray[np.datetime64] | None      
    _stacked: xr.Dataset              # lazy xr dataset        
    _xyz: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    @property
    def xyz(self) -> NDArray[np.float64]:
        """Cartesian 3D coordinates of the nodes, computed once and cached."""
        if self._xyz is None:
            self._xyz = lonlat_to_xyz(self.lon, self.lat)
        return self._xyz

    @classmethod
    def from_grid(cls, ds: xr.Dataset) -> "FlatGrid":
        """Flatten a grid (2D or 3D) into a node cloud. Values stay lazy."""
        stacked = ds.stack(node=("lat", "lon"))
        time = ds["time"].values if "time" in ds.coords else None
        return cls(lon=stacked["lon"].values, lat=stacked["lat"].values,
                   time=time, _stacked=stacked)

    def read_values(self, node_idx: ArrayLike,
                    tsel_idx: ArrayLike | None = None) -> dict[str, NDArray[np.float64]]:
        """Read variable values ONLY at the retained (node[, time]) indices"""
        node = xr.DataArray(np.asarray(node_idx), dims="pts")
        out = {}
        for var in self._stacked.data_vars: # _stacked is lazy ds with only lon/lat/time in memory
            da_var = self._stacked[var]
            if tsel_idx is None:
                sel = da_var.isel(node=node)
            else:
                sel = da_var.isel(node=node, time=xr.DataArray(np.asarray(tsel_idx), dims="pts"))
            out[str(var)] = sel.values
        return out
