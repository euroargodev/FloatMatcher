# flatgrid.py: the reference point cloud (flattened grid), internal to nearest,
# with the spatial and temporal nearest lookups (KDTree) on it

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from .constants import REF_TIME, TIME_UNIT
from .geo import lonlat_to_xyz
from .pointset import PointSet


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
        return cls(lon=stacked["lon"].values, lat=stacked["lat"].values, # trigger memory import xr dataset 
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

    def spatial_nearest(self, points: PointSet,
                        k: int = 1) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Nearest nodes of the points in space

        Parameters
        ----------
        points: :class:`PointSet`
        k: int, default=1
            Number of neighbours.

        Returns
        -------
        tuple of arrays
            Distances (km) and indices of the nodes.
        """
        dist: NDArray[np.float64]
        idx: NDArray[np.int64]
        dist, idx = cKDTree(self.xyz).query(points.xyz, k=k)
        return dist, idx

    def temporal_nearest(self, points: PointSet,
                         k: int = 1) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Nearest time steps of the points

        Parameters
        ----------
        points: :class:`PointSet`
        k: int, default=1
            Number of neighbours.

        Returns
        -------
        tuple of arrays
            Time gaps (seconds) and indices of the time steps.

        Raises
        ------
        ValueError
            If the grid or the points have no time.
        """
        time_delta: NDArray[np.float64]
        idx: NDArray[np.int64]
        if self.time is None:
            raise ValueError("the grid has no time axis")
        if points.time is None:
            raise ValueError("points have no time: they cannot be matched on a grid with a time axis")
        grid_tree = cKDTree(_to_seconds(self.time)[:, None])
        time_delta, idx = grid_tree.query(_to_seconds(points.time)[:, None], k=k)
        return time_delta, idx


def _to_seconds(times: NDArray[np.datetime64]) -> NDArray[np.float64]:
    """Convert datetime64 to floating-point seconds since a fixed epoch.

    Working in a common float unit lets the 1D KDTree measure time distance,
    and returning *seconds* makes the time_gap_seconds constraint directly comparable.
    """
    delta = np.asarray(times, dtype=f"datetime64[{TIME_UNIT}]") - REF_TIME
    seconds: NDArray[np.float64] = delta / np.timedelta64(1, "s")
    return seconds
