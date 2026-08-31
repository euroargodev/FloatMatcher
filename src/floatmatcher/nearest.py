# nearest.py: nearest-neighbor matchup via separated spatial/temporal KDTrees.
#
# The spatial half is prepared ONCE (prepare) and reused on every temporal
# packet (match_packet), because the grid geometry is identical across packets.
# The packet loop itself lives in the orchestrator; this module provides the
# two halves and a single-pass `match` for the non-batched case.

import numpy as np
from numpy.typing import NDArray

from .method import MatchupMethod, Constraints
from .gridset import GridSet
from .pointset import PointSet
from .results import MatchupResult
from .flatgrid import FlatGrid
from .index import spatial_index, temporal_index

class NearestNeighbor():
    """Nearest-neighbor colocalization.

    ``max_files`` is the batching granularity: it caps how many files are opened
    simultaneously per packet, bounding open_mfdataset I/O. The orchestrator
    reads it to size the packets; the packet loop lives in the orchestrator.
    """

    def __init__(self, max_dist_km: int = 25, 
                 max_time: np.timedelta64 = np.timedelta64(1, "D"),
                 k_nearest : int = 1) -> None :
        self.max_dist_km = max_dist_km
        self.max_time = max_time
        self.k_nearest = k_nearest

    @property
    def max_time_seconds(self) -> float:
        return float(self.max_time / np.timedelta64(1, "s"))

    # def prepare(self, grid: GridSet, points: PointSet,
    #             constraints: Constraints) -> _PreparedNearest:
    #     """Build the spatial half once (tree + spatial query), from a packet."""
    #     reference = grid_to_reference(grid.dataset)
    #     dist_km, spatial_idx = SpatialIndex(reference.xyz).query(points)
    #     return _PreparedNearest(points, constraints, dist_km, spatial_idx,
    #                             _geometry_fingerprint(reference))

    # def match(self, grid: GridSet, points: PointSet,
    #           constraints: Constraints) -> MatchupResult:
    #     """Single grid (2D, or one packet): prepare + match in one shot.

    #     Non-batched entry point, output identical to before. The batched path
    #     prepares once and calls match_packet per packet from the orchestrator.
    #     """
    #     return self.prepare(grid, points, constraints).match_packet(grid)