# index.py: spatial + temporal KDTree lookup for a reference grid

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .pointset import PointSet
from .reference import ReferenceSet
from .constants import REF_TIME, TIME_UNIT


def _to_days(times: NDArray[np.datetime64]) -> NDArray[np.float64]:
    """Convert datetime64 to floating-point days since a fixed epoch.

    Working in a common float unit lets the 1D KDTree measure time distance,
    and returning *days* makes the max_time_days constraint directly comparable.
    """
    delta = np.asarray(times, dtype=f"datetime64[{TIME_UNIT}]") - REF_TIME
    return delta / np.timedelta64(1, "D")      # nanoseconds → days, as float


class GridIndex:
    """Nearest-neighbor lookup over a reference grid, space and time separated.

    Two independent trees, which is the right model for a regular grid: the
    spatial positions repeat at every time step, so 'closest in space' and
    'closest in time' are independent questions, combined afterwards.

    The spatial tree is built once at construction. The temporal tree is built
    only if the reference has a time axis (3D grids).
    """

    def __init__(self, reference: ReferenceSet):
        self.reference = reference
        self._spatial_tree = cKDTree(reference.xyz)

        if reference.time is not None:
            self._grid_days = _to_days(reference.time)
            self._time_tree = cKDTree(self._grid_days[:, None])   # 1D → (N, 1)
        else:
            self._grid_days = None
            self._time_tree = None

    def query_spatial(self, points: PointSet) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """For each point, distance (km) and index of the nearest grid node."""
        dist_km, idx = self._spatial_tree.query(points.xyz)
        return dist_km, idx

    def query_temporal(self, points: PointSet) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """For each point, |time difference| (days) and index of nearest time step.

        Only valid on a 3D reference; raises on a 2D one.
        """
        if self._time_tree is None:
            raise ValueError("query_temporal called on a 2D reference (no time axis)")
        point_days = _to_days(points.time)
        dist_days, idx = self._time_tree.query(point_days[:, None])
        return dist_days, idx