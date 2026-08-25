# nearest.py: nearest-neighbor matchup via separated spatial/temporal KDTrees

import numpy as np

from .method import MatchupMethod, Constraints
from .gridset import GridSet
from .pointset import PointSet
from .results import MatchupResult
from .reference import grid_to_reference
from .index import GridIndex


class NearestNeighbor(MatchupMethod):
    """Nearest-neighbor colocalization."""

    def match(self, grid: GridSet, points: PointSet,
              constraints: Constraints) -> MatchupResult:
        reference = grid_to_reference(grid.dataset)
        index = GridIndex(reference)
        n = len(points.lon)

        # --- spatial nearest neighbor (always) ---
        dist_km, spatial_idx = index.query_spatial(points)
        valid = dist_km <= constraints.max_dist_km

        # --- temporal nearest neighbor (3D grids only) ---
        if grid.regime == "3D":
            time_delta, temporal_idx = index.query_temporal(points)
            valid = valid & (time_delta <= constraints.max_time_days)
        else:
            time_delta = np.full(n, np.nan)
            temporal_idx = None

        # --- read the retained ONLY at the retained nodes for each variable, then mask invalids ---
        picked = reference.read_values(spatial_idx, temporal_idx)
        values = {}
        for var, vals in picked.items():
            values[var] = np.where(valid, vals, np.nan)

        # invalid points carry no meaningful distance/time either
        dist_km = np.where(valid, dist_km, np.nan)
        time_delta = np.where(valid, time_delta, np.nan)

        return MatchupResult(values=values, distance_km=dist_km,
                             time_delta=time_delta, valid=valid)
