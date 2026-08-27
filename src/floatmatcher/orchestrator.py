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
from typing import Any

import xarray as xr

from .gridset import GridSet
from .local_source import LocalSource
from .method import Constraints, MatchupMethod
from .nearest import NearestNeighbor
from .pointset import PointSet
from .results import MatchupResult

_Source = LocalSource | str | Sequence[str]
_Variables = str | Sequence[str] | None


class Orchestrator:
    """Configure once (product + method + constraints), colocalize many times.

    Parameters
    ----------
    product : object exposing COORD_MAP, LON_RANGE and normalize(raw_ds)
    method  : a MatchupMethod (NearestNeighbor, Interpolation, ...)
    constraints : Constraints, optional
    """

    def __init__(self, product: Any, method: MatchupMethod,
                 constraints: Constraints | None = None) -> None:
        self.product = product
        self.method = method
        self.constraints = constraints or Constraints()

    def colocalize(self, source: _Source, points: PointSet,
                   variables: _Variables = None) -> MatchupResult:
        """Colocalize `points` (a PointSet) against a grid `source`.

        `source` is a LocalSource, or a file path / list of paths (wrapped in a
        LocalSource with an ExplicitFiles resolver for convenience).
        Returns a MatchupResult.
        """
        local = self._as_local_source(source)
        if isinstance(self.method, NearestNeighbor):
            return self._colocalize_nearest(local, points, variables)
        return self._colocalize_interp(local, points, variables)

    # ── per-method strategies ────────────────────────────────────────────

    def _colocalize_nearest(self, local: LocalSource, points: PointSet,
                            variables: _Variables) -> MatchupResult:
        """Nearest with temporal batching.

        Open files in packets of at most `method.max_files` (bounding I/O),
        build the spatial index once from the first packet, match every packet,
        and keep each point's best match across packets via update_best.
        """
        assert isinstance(self.method, NearestNeighbor)
        files = local.resolver.files_for(points)
        max_files = self.method.max_files
        chunks = [files[i:i + max_files] for i in range(0, len(files), max_files)]

        prepared = None
        result = None
        for chunk in chunks:
            grid = self._open_chunk_grid(local, chunk, variables)
            if prepared is None:                     # spatial built once, reused
                prepared = self.method.prepare(grid, points, self.constraints)
            partial = prepared.match_packet(grid)
            result = partial if result is None else result.update_best(partial)
        assert result is not None
        return result

    def _colocalize_interp(self, local: LocalSource, points: PointSet,
                           variables: _Variables) -> MatchupResult:
        """Interpolation: single open for now.

        RAM-slice batching (a different strategy: temporal slices bounded by a
        memory budget, with one-slice overlap) will slot in here later, without
        touching the nearest path.
        """
        files = local.resolver.files_for(points)
        grid = self._open_chunk_grid(local, files, variables)
        return self.method.match(grid, points, self.constraints)

    # ── shared helpers ───────────────────────────────────────────────────

    def _open_chunk_grid(self, local: LocalSource, paths: Sequence[str],
                         variables: _Variables) -> GridSet:
        """Open one packet of files -> normalized, validated GridSet."""
        raw = local.open_paths(paths)
        ds = self._select_variables(self.product.normalize(raw), variables)
        return GridSet(ds, declared_lon_range=self.product.LON_RANGE)

    @staticmethod
    def _select_variables(ds: xr.Dataset, variables: _Variables) -> xr.Dataset:
        """Keep only the requested variables. None -> keep all.

        Raises a clear error (listing what's available) if a name is unknown,
        rather than xarray's terse KeyError.
        """
        if variables is None:
            return ds
        wanted = [variables] if isinstance(variables, str) else list(variables)
        missing = [v for v in wanted if v not in ds.data_vars]
        if missing:
            raise ValueError(
                f"variables not found in the grid: {missing}. "
                f"Available: {list(ds.data_vars)}."
            )
        return ds[wanted]

    @staticmethod
    def _as_local_source(source: _Source) -> LocalSource:
        """Accept a ready LocalSource, or a path / list of paths for convenience."""
        if isinstance(source, LocalSource):
            return source
        return LocalSource.from_paths(source)