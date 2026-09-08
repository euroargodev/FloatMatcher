# orchestrator.py: the top-level coordinator (public API of the library).


from collections.abc import Sequence

import numpy as np 

from .gridset import GridSet
from .matchup import NearestNeighbor
# from .interpolation import Interpolation
from .pointset import PointSet
from .results import MatchupResult
from .products import Product
from .flatgrid import FlatGrid
from .utils import _select_variables
from .neighbors import spatial_nearest, temporal_nearest

class Orchestrator:
    """Orchestrator gather all Points Pointset(), variables needed, setup Product(). 
    It prepares all elements to call the method in .match() function. 
    All constraints and parametrization lives in .match()
    """

    def __init__(self, points: PointSet, 
                 variables: str | list[str] | None, product: Product) -> None:
        self.points = points
        self.variables = variables
        self.product = product
        self._files: list[str] | None = None

    @property
    def files(self) -> list[str]:
        if self._files is None:
            # always give points just not used in case of ExplicitFiles resolver
            self._files = self.product.files_for(self.points)
        return self._files

    def match(self, method: NearestNeighbor) -> MatchupResult:
        if isinstance(method, NearestNeighbor):
            return self._match_nearest(method)
        # elif isinstance(method, Interpolation):
        #     return self._match_interp()
        else:
            raise AttributeError("unknown method")

    def _match_nearest(self, method: NearestNeighbor) -> MatchupResult:
        print("matchin nearest method : no batching method yet")
        files_to_process = self.files # property is only trigger when called the first time
        grid_full = self._open_lazy_grid(files_to_process) # GridSet object

        # starting by lonlat2xy on spatial grid
        flat_grid = FlatGrid.from_grid(grid_full.dataset) # return FlatGrid object
        grid_stacked = flat_grid.xyz

        # starting Nearest method : apply kdtree on spatial 
        dist_km, spatial_idx = spatial_nearest(grid_stacked, self.points, k=method.k_nearest)
        valid_spatial = dist_km <= method.max_dist_km

        idx_count = len(self.points.lon)
        if grid_full.regime == "3D":
            assert flat_grid.time is not None
            time_delta, temporal_idx = temporal_nearest(flat_grid.time, 
                                                        self.points, 
                                                        k=method.k_nearest
                                                        )
            valid = valid_spatial & (time_delta <= method.max_time_seconds)
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
                                time_delta=time_delta_out, valid=valid, points=self.points)


    def _open_lazy_grid(self, paths: Sequence[str]) -> GridSet:
        """Open set of files -> normalized, variable selected and validated GridSet."""
        ds_raw = self.product.open_paths(paths)
        ds = _select_variables(self.product.normalize(ds_raw), self.variables)
        return GridSet(ds)
    
