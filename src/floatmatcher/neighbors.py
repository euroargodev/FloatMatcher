# index.py: spatial and temporal KDTree lookups over a reference grid.
#
# Spatial and temporal are SEPARATED because they now have different lifetimes:
#   - SpatialIndex is built ONCE (grid geometry is identical on every packet);
#   - TemporalIndex is built PER packet (only the time axis changes).
# On a regular grid the spatial positions repeat at every time step, so
# "closest in space" and "closest in time" are independent questions.

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .pointset import PointSet
from .constants import REF_TIME, TIME_UNIT


def _to_seconds(times: NDArray[np.datetime64]) -> NDArray[np.float64]:
    """Convert datetime64 to floating-point seconds since a fixed epoch.

    Working in a common float unit lets the 1D KDTree measure time distance,
    and returning *seconds* makes the max_time_seconds constraint directly comparable.
    """
    delta = np.asarray(times, dtype=f"datetime64[{TIME_UNIT}]") - REF_TIME
    seconds: NDArray[np.float64] = delta / np.timedelta64(1, "s")
    return seconds


def spatial_nearest(grid_xyz: NDArray[np.float64], points: PointSet,
                    k: int = 1) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    dist: NDArray[np.float64]
    idx: NDArray[np.int64]
    dist, idx = cKDTree(grid_xyz).query(points.xyz, k=k)
    return dist, idx
    
def temporal_nearest(grid_times: NDArray[np.datetime64], points: PointSet,
                     k: int = 1) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    time_delta: NDArray[np.float64]
    idx: NDArray[np.int64]
    grid_tree = cKDTree(_to_seconds(grid_times)[:, None])
    time_delta, idx = grid_tree.query(_to_seconds(points.time)[:, None], k=k)
    return time_delta, idx

