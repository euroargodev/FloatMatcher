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
from .reference import grid_to_reference, ReferenceSet
from .index import SpatialIndex, TemporalIndex

_Geometry = tuple[tuple[int, ...], float, float, float, float]


def _geometry_fingerprint(reference: ReferenceSet) -> _Geometry:
    """Cheap signature of a grid's spatial geometry.

    Used to assert the geometry stays constant across packets -- the invariant
    that lets the spatial index be built once. Compares shape and the endpoint
    lon/lat values rather than the full arrays (light, per-packet check).
    """
    lon, lat = reference.lon, reference.lat
    return (lon.shape, float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1]))


class _PreparedNearest:
    """Spatial half of a nearest matchup, computed once and reused per packet.

    Holds only spatial arrays (no packet data), so a single instance drives the
    whole packet loop. `match_packet` adds the packet's temporal half and reads
    the values, confronting ALL points with the packet (boundaries are handled
    later by MatchupResult.update_best).
    """

    def __init__(self, points: PointSet, constraints: Constraints,
                 dist_km: NDArray[np.float64], spatial_idx: NDArray[np.int64],
                 geometry: _Geometry) -> None:
        self.points = points
        self.constraints = constraints
        self.dist_km = dist_km
        self.spatial_idx = spatial_idx
        self.valid_spatial = dist_km <= constraints.max_dist_km
        self._geometry = geometry

    def match_packet(self, grid: GridSet) -> MatchupResult:
        """Colocalize all points against ONE packet (one open grid)."""
        reference = grid_to_reference(grid.dataset)
        if _geometry_fingerprint(reference) != self._geometry:
            raise ValueError(
                "nearest: grid geometry changed between packets; the spatial "
                "index cannot be reused. Packets must share the same lat/lon."
            )

        n = len(self.points.lon)
        if grid.regime == "3D":
            assert reference.time is not None
            time_delta, temporal_idx = TemporalIndex(reference.time).query(self.points)
            valid = self.valid_spatial & (time_delta <= self.constraints.max_time_days)
        else:
            time_delta = np.full(n, np.nan)
            temporal_idx = None
            valid = self.valid_spatial

        # read ONLY at valid points: no wasted read for out-of-window points
        idx = np.where(valid)[0]
        t_sel = None if temporal_idx is None else temporal_idx[idx]
        picked = reference.read_values(self.spatial_idx[idx], t_sel)

        # scatter each variable's valid values back to full point-length arrays
        values = {}
        for var, vals in picked.items():
            full = np.full(n, np.nan)
            full[idx] = vals
            values[var] = full

        # invalid points carry no meaningful distance/time either
        dist_km = np.full(n, np.nan)
        dist_km[idx] = self.dist_km[idx]
        time_delta_out = np.full(n, np.nan)
        time_delta_out[idx] = time_delta[idx]

        return MatchupResult(values=values, distance_km=dist_km,
                             time_delta=time_delta_out, valid=valid)


class NearestNeighbor(MatchupMethod):
    """Nearest-neighbor colocalization.

    ``max_files`` is the batching granularity: it caps how many files are opened
    simultaneously per packet, bounding open_mfdataset I/O. The orchestrator
    reads it to size the packets; the packet loop lives in the orchestrator.
    """

    def __init__(self, max_files: int = 100):
        self.max_files = max_files

    def prepare(self, grid: GridSet, points: PointSet,
                constraints: Constraints) -> _PreparedNearest:
        """Build the spatial half once (tree + spatial query), from a packet."""
        reference = grid_to_reference(grid.dataset)
        dist_km, spatial_idx = SpatialIndex(reference.xyz).query(points)
        return _PreparedNearest(points, constraints, dist_km, spatial_idx,
                                _geometry_fingerprint(reference))

    def match(self, grid: GridSet, points: PointSet,
              constraints: Constraints) -> MatchupResult:
        """Single grid (2D, or one packet): prepare + match in one shot.

        Non-batched entry point, output identical to before. The batched path
        prepares once and calls match_packet per packet from the orchestrator.
        """
        return self.prepare(grid, points, constraints).match_packet(grid)