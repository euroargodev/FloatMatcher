# orchestrator.py: the top-level coordinator (public API of the library).
#
# Ties the pieces together and hides the plumbing (access -> normalize ->
# GridSet -> match). It does not open files itself: file access lives in the
# LocalSource layer, which the orchestrator invokes.
#
# Each method has its own temporal-batching strategy, so the orchestrator holds
# one sub-method per method (nearest / interp) rather than a single generic
# loop. The public `colocalize` dispatches to the right one.

from collections.abc import Sequence

import xarray as xr
import numpy as np 
from numpy.typing import NDArray, ArrayLike


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

    def __init__(self, points: PointSet, variables: str | list[str] | None, product: Product) -> None:
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

    def match(self, method) -> MatchupResult:
        if isinstance(method, NearestNeighbor):
            return self._match_nearest(method)
        # elif isinstance(method, Interpolation):
        #     return self._match_interp()
        else:
            raise AttributeError("unknown method")

    def _match_nearest(self, method: NearestNeighbor):
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
            time_delta, temporal_idx = temporal_nearest(flat_grid.time, self.points, k=method.k_nearest)
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
            tsel_idx = temporal_idx[idx]

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
                                time_delta=time_delta_out, valid=valid)


    def _open_lazy_grid(self, paths) -> GridSet:
        """Open set of files -> normalized, variable selected and validated GridSet."""
        ds_raw = self.product.open_paths(paths)
        ds = _select_variables(self.product.normalize(ds_raw), self.variables)
        return GridSet(ds)
    

    # def colocalize(self, source: _Source, points: PointSet,
    #                variables: _Variables = None) -> MatchupResult:
    #     """Colocalize `points` (a PointSet) against a grid `source`.

    #     `source` is a LocalSource, or a file path / list of paths (wrapped in a
    #     LocalSource with an ExplicitFiles resolver for convenience).
    #     Returns a MatchupResult.
    #     """
    #     local = self._as_local_source(source)
    #     if isinstance(self.method, NearestNeighbor):
    #         return self._colocalize_nearest(local, points, variables)
    #     elif isinstance(self.method, Interpolation):
    #         return self._colocalize_interp(local, points, variables)
    #     else:
    #         raise AttributeError("unknown method")

    # # ── per-method strategies ────────────────────────────────────────────

    # def _colocalize_nearest(self, local: LocalSource, points: PointSet,
    #                         variables: _Variables) -> MatchupResult:
    #     """Nearest with temporal batching.

    #     Open files in packets of at most `method.max_files` (bounding I/O),
    #     build the spatial index once from the first packet, match every packet,
    #     and keep each point's best match across packets via update_best.
    #     """
    #     assert isinstance(self.method, NearestNeighbor)
    #     files = local.resolver.files_for(points)
    #     max_files = self.method.max_files
    #     chunks = [files[i:i + max_files] for i in range(0, len(files), max_files)]

    #     prepared = None
    #     result = None
    #     for chunk in chunks:
    #         grid = self._open_chunk_grid(local, chunk, variables)
    #         if prepared is None:                     # spatial built once, reused
    #             prepared = self.method.prepare(grid, points, self.constraints)
    #         partial = prepared.match_packet(grid)
    #         result = partial if result is None else result.update_best(partial)
    #     assert result is not None
    #     return result

    # def _colocalize_interp(self, local: LocalSource, points: PointSet,
    #                        variables: _Variables) -> MatchupResult:
    #     """Interpolation: single open for now.

    #     RAM-slice batching (a different strategy: temporal slices bounded by a
    #     memory budget, with one-slice overlap) will slot in here later, without
    #     touching the nearest path.
    #     """
    #     files = local.resolver.files_for(points)
    #     grid = self._open_chunk_grid(local, files, variables)
    #     return self.method.match(grid, points, self.constraints)

    # # ── shared helpers ───────────────────────────────────────────────────

    # def _open_chunk_grid(self, local: LocalSource, paths: Sequence[str],
    #                      variables: _Variables) -> GridSet:
    #     """Open one packet of files -> normalized, validated GridSet."""
    #     raw = local.open_paths(paths)
    #     ds = self._select_variables(self.product.normalize(raw), variables)
    #     return GridSet(ds, declared_lon_range=self.product.LON_RANGE)

    # @staticmethod
    # def _select_variables(ds: xr.Dataset, variables: _Variables) -> xr.Dataset:
    #     """Keep only the requested variables. None -> keep all.

    #     Raises a clear error (listing what's available) if a name is unknown,
    #     rather than xarray's terse KeyError.
    #     """
    #     if variables is None:
    #         return ds
    #     wanted = [variables] if isinstance(variables, str) else list(variables)
    #     missing = [v for v in wanted if v not in ds.data_vars]
    #     if missing:
    #         raise ValueError(
    #             f"variables not found in the grid: {missing}. "
    #             f"Available: {list(ds.data_vars)}."
    #         )
    #     return ds[wanted]

    # @staticmethod
    # def _as_local_source(source: _Source) -> LocalSource:
    #     """Accept a ready LocalSource, or a path / list of paths for convenience."""
    #     if isinstance(source, LocalSource):
    #         return source
    #     return LocalSource.from_paths(source)