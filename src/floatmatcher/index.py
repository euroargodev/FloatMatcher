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


def _to_days(times: NDArray[np.datetime64]) -> NDArray[np.float64]:
    """Convert datetime64 to floating-point days since a fixed epoch.

    Working in a common float unit lets the 1D KDTree measure time distance,
    and returning *days* makes the max_time_days constraint directly comparable.
    """
    delta = np.asarray(times, dtype=f"datetime64[{TIME_UNIT}]") - REF_TIME
    days: NDArray[np.float64] = delta / np.timedelta64(1, "D")
    return days


class SpatialIndex:
    """Nearest-neighbor lookup over grid node positions (built once, reused).

    The grid geometry (lat/lon) does not change from one temporal packet to the
    next, so the spatial answer (nearest node + distance) is packet-independent:
    this index is built a single time and queried on every packet.
    """

    def __init__(self, xyz: NDArray[np.float64]) -> None:
        self._tree = cKDTree(xyz)

    def query(self, points: PointSet) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """For each point, distance (km) and index of the nearest grid node."""
        dist: NDArray[np.float64]
        idx: NDArray[np.int64]
        dist, idx = self._tree.query(points.xyz)
        return dist, idx


class TemporalIndex:
    """Nearest-neighbor lookup over one packet's time axis (built per packet)."""

    def __init__(self, times: NDArray[np.datetime64]) -> None:
        self._days = _to_days(times)
        self._tree = cKDTree(self._days[:, None])          # 1D -> (N, 1)

    def query(self, points: PointSet) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """For each point, |time difference| (days) and index of nearest step."""
        point_days = _to_days(points.time)
        dist: NDArray[np.float64]
        idx: NDArray[np.int64]
        dist, idx = self._tree.query(point_days[:, None])
        return dist, idx