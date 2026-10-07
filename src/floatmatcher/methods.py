# methods.py: matchup methods.

# The spatial half is prepared ONCE (prepare) and reused on every temporal
# packet (match_packet), because the grid geometry is identical across packets.
# The packet loop itself lives in the FloatMatcher; this module provides the
# two halves and a single-pass `match` for the non-batched case.

from abc import ABC, abstractmethod

import numpy as np

from .flatgrid import FlatGrid
from .gridset import GridSet
from .matchup_results import MatchupResult
from .neighbors import spatial_nearest, temporal_nearest
from .pointset import PointSet
from .utils import indented_repr


class Method(ABC):
    """Contract shared by every matchup method."""

    def __repr__(self) -> str:
        return indented_repr(self)

    @abstractmethod
    def apply(self, grid: GridSet, points: PointSet) -> MatchupResult:
        ...



class NearestNeighbor(Method):
    """Nearest-neighbor matchup method"""

    def __init__(self, radius: int = 25,
                 time_gap: np.timedelta64 = np.timedelta64(1, "D"),
                 k_nearest : int = 1) -> None :
        self.radius = radius
        self.time_gap = time_gap
        self.k_nearest = k_nearest

    @property
    def time_gap_seconds(self) -> float:
        return float(self.time_gap / np.timedelta64(1, "s"))

    def apply(self, grid: GridSet, points: PointSet) -> MatchupResult:
        # starting by lonlat2xy on spatial grid
        # return FlatGrid object, flatten grid
        flat_grid = FlatGrid.from_grid(grid.dataset)
        # convert into carthesian coordinates
        grid_stacked = flat_grid.xyz

        # starting Nearest method : apply kdtree on spatial
        dist_km, spatial_idx = spatial_nearest(grid_stacked, points, k=self.k_nearest)
        valid_spatial = dist_km <= self.radius

        idx_count = len(points.lon)
        if grid.regime == "3D":
            assert flat_grid.time is not None
            time_delta, temporal_idx = temporal_nearest(flat_grid.time,
                                                        points,
                                                        k=self.k_nearest
                                                        )
            valid = valid_spatial & (time_delta <= self.time_gap_seconds)
        else:
            time_delta = np.full(idx_count, np.nan)
            temporal_idx = None
            valid = valid_spatial

        # read ONLY at valid points: no wasted read for out-of-window points
        # select indexes of spatial and time
        idx = np.where(valid)[0]
        node_idx = spatial_idx[idx]       # grid node index of retained points
        tsel_idx = None              # set temporal case
        if temporal_idx is not None:
            tsel_idx = temporal_idx[idx] # apply on every node. filtering is made after (costless)

        # retreive data only at good positions : select in dataset stacked of FlatGrid object
        picked = flat_grid.read_values(node_idx, tsel_idx)

        # scatter each variable's valid values back to full PointSet-length.
        # picked is from _stacked which is flatten, not PointSet lenght :)
        values = {}
        for var, vals in picked.items():
            full = np.full(idx_count, np.nan)
            full[idx] = vals
            values[var] = full

        # invalid points carry no meaningful distance/time either
        dist_out = np.full(idx_count, np.nan)
        dist_out[idx] = dist_km[idx]
        time_delta_out = np.full(idx_count, np.nan)
        time_delta_out[idx] = time_delta[idx]

        return MatchupResult(values=values, distance_km=dist_out,
                                time_delta=time_delta_out, valid=valid, points=points)


# key given to FloatMatcher.set_method(type=...) -> method class
available_methods: dict[str, type[Method]] = {
    "nearest": NearestNeighbor,
}
