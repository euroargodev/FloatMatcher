# orchestrator.py: the top-level coordinator (public API of the library).
#
# Ties the pieces together and hides the plumbing (access -> normalize ->
# GridSet -> match). It no longer opens files itself: file access lives in the
# LocalSource layer (§4.3), so the orchestrator only coordinates.
#
# MINIMAL VERSION: single pass, no batching, no cache. Temporal batching will
# slot in between "open" and "match" (split into packets, reinject via
# origin_index) — that is what the config carried here will drive.

from .gridset import GridSet
from .local_source import LocalSource, ExplicitFiles
from .method import Constraints


class Orchestrator:
    """Configure once (product + method + constraints), colocalize many times.

    Parameters
    ----------
    product : object exposing COORD_MAP, LON_RANGE and normalize(raw_ds)
    method  : a MatchupMethod (NearestNeighbor, Interpolation, ...)
    constraints : Constraints, optional
    """

    def __init__(self, product, method, constraints=None):
        self.product = product
        self.method = method
        self.constraints = constraints or Constraints()

    def colocalize(self, source, points, variables=None):
        """Colocalize `points` (a PointSet) against a grid `source`.

        `source` is a LocalSource, or a file path / list of paths (wrapped in a
        LocalSource with an ExplicitFiles resolver for convenience).
        Returns a MatchupResult.
        """
        grid = self._build_grid(source, points, variables)
        return self.method.match(grid, points, self.constraints)

    def _build_grid(self, source, points, variables):
        """Open (via LocalSource), normalize, and wrap into a validated GridSet."""
        local = self._as_local_source(source)
        raw = local.open(points)
        normalized = self.product.normalize(raw)
        normalized = self._select_variables(normalized, variables)
        return GridSet(normalized, declared_lon_range=self.product.LON_RANGE)
    
    @staticmethod
    def _select_variables(ds, variables):
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
    def _as_local_source(source):
        """Accept a ready LocalSource, or a path / list of paths for convenience."""
        if isinstance(source, LocalSource):
            return source
        return LocalSource(ExplicitFiles(source))