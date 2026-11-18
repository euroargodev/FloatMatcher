# methods.py: matchup methods (contract and registry).
#
# The grid is opened ONCE by the product and handed to apply(): the spatial and temporal
# lookups run on the whole grid in a single pass (no temporal packets), and the values are
# read lazily, only at the valid points.

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .flatgrid import FlatGrid
from .matchup_results import MatchupResult
from .pointset import PointSet
from .utils import _indented_repr, _signature_defaults
from .weights import available_kernels

class Method(ABC):
    """Prototype of the matchup methods

    A method holds its parameters and matches points with a grid in ``apply``. It knows
    nothing of the product or of :class:`.FloatMatcher`: it receives the opened grid and the
    points. Subclasses implement ``apply`` and are registered in ``available_methods``.

    Attributes
    ----------
    required_params: set of str
        Parameters of the method that must be given to ``set_method``.
    optional_params: set of str
        Parameters of the method that have a default value, given to ``set_method`` to
        change it. The default values are read in the signature of ``__init__``.
    """

    radius: int
    time_gap: np.timedelta64
    # parameters given to set_method
    required_params: set[str] = set()
    optional_params: set[str] = set()

    def __repr__(self) -> str:
        return _indented_repr(self)

    @classmethod
    def _optional_params(cls) -> dict[str, Any]:
        """Optional parameters of the method -> their default value"""
        return _signature_defaults(cls, cls.optional_params)

    @property
    def time_gap_seconds(self) -> float:
        """``time_gap`` in seconds"""
        return float(self.time_gap / np.timedelta64(1, "s"))

    def _neighbors(self, flat_grid: FlatGrid, points: PointSet, k_nearest: int,
                   n_frames: int) -> tuple[NDArray, NDArray, NDArray,
                                           NDArray, NDArray | None, NDArray]:
        """k nearest nodes and n nearest time steps of the points, sorted from the nearest

        Missing neighbors (more asked than in the grid) have NaN distance or time gap and are
        invalid. Their index is set to 0, so that it can be read, their weight being 0.

        Returns
        -------
        tuple
            dist_km (n, K), spatial_idx (n, K), valid_spatial (n, K),
            time_delta (n, N), temporal_idx (n, N) or None, valid_time (n, N).
            Without time axis, N is 1, time_delta is NaN and valid_time is True.
        """
        idx_count = len(points.lon)

        # k nearest nodes in space, sorted from the nearest: (n, K)
        dist_km, spatial_idx = flat_grid.spatial_nearest(points, k=k_nearest)
        dist_km = dist_km.reshape(idx_count, -1)
        spatial_idx = spatial_idx.reshape(idx_count, -1)
        # missing neighbors (k larger than the grid) come back as inf: not found
        missing = np.isinf(dist_km)
        dist_km[missing] = np.nan
        # placeholder index, never used: the neighbor is invalid and valid set to false
        spatial_idx[missing] = 0
        valid_spatial = dist_km <= self.radius             # NaN gives False

        # n nearest time steps, sorted from the nearest: (n, N)
        temporal_idx: NDArray | None
        if flat_grid.time is not None:
            time_delta, temporal_idx = flat_grid.temporal_nearest(points, k=n_frames)
            time_delta = time_delta.reshape(idx_count, -1)
            temporal_idx = temporal_idx.reshape(idx_count, -1)
            missing = np.isinf(time_delta)
            time_delta[missing] = np.nan
            # placeholder index, never used: the neighbor is invalid and valid set to false
            temporal_idx[missing] = 0
            valid_time = time_delta <= self.time_gap_seconds
        else:
            # grid without time axis: n_frames ignored
            time_delta = np.full((idx_count, 1), np.nan)
            temporal_idx = None
            valid_time = np.ones((idx_count, 1), dtype=bool)

        return dist_km, spatial_idx, valid_spatial, time_delta, temporal_idx, valid_time

    @abstractmethod
    def apply(self, dataset: xr.Dataset, points: PointSet) -> MatchupResult:
        """Match the points with a grid

        Parameters
        ----------
        dataset: xarray.Dataset
            Grid with the ``lon`` and ``lat`` coordinates, and ``time`` if it has a time axis.
        points: :class:`.PointSet`

        Returns
        -------
        :class:`.MatchupResult`
        """
        ...



class NearestNeighbor(Method):
    """Nearest-neighbor matchup method

    Each point takes the values of the nearest node of the grid, and of the nearest time step
    if the grid has a time axis. Points beyond ``radius`` or ``time_gap`` are invalid.

    Parameters
    ----------
    radius: int, default=25
        Maximum distance to the node, in km (straight line on the sphere).
    time_gap: numpy.timedelta64, default=1 day
        Maximum gap to the time step. Ignored for a grid without time axis.
    """
    optional_params = {"radius", "time_gap", "k_nearest", "n_frames"}

    def __init__(self, radius: int = 25,
                 time_gap: np.timedelta64 = np.timedelta64(1, "D"),
                 k_nearest: int = 1,
                 n_frames: int = 1) -> None :
        self.radius = radius
        self.time_gap = time_gap
        self.k_nearest = k_nearest
        self.n_frames = n_frames

    def apply(self, dataset: xr.Dataset, points: PointSet) -> MatchupResult:
        """Match the points with the nearest node of the grid

        Invalid points have NaN values, distance and time gap in the result. Values are read
        only for the valid points.

        Parameters
        ----------
        dataset: xarray.Dataset
            Grid with the ``lon`` and ``lat`` coordinates, and ``time`` if it has a time axis.
        points: :class:`.PointSet`

        Returns
        -------
        :class:`.MatchupResult`

        Raises
        ------
        ValueError
            If the grid has a time axis and the points have no time.
        """
        # starting by lonlat2xy on spatial grid
        # return FlatGrid object, flatten grid
        flat_grid = FlatGrid.from_grid(dataset)

        # k nearest nodes and n nearest time steps, sorted from the nearest
        (dist_km, spatial_idx, valid_spatial,
         time_delta, temporal_idx, valid_time) = self._neighbors(flat_grid, points,
                                                                 self.k_nearest, self.n_frames)

        # one neighbor = one node x one time step: (n, K, N)
        valid = valid_spatial[:, :, None] & valid_time[:, None, :]

        # read ONLY the valid neighbors, as (node, time step) pairs:
        # no wasted read for out-of-window neighbors
        idx = np.where(valid)                        # (point, node, time step) of the valid neighbors
        node_idx = spatial_idx[idx[0], idx[1]]       # grid node index of retained neighbors
        tsel_idx = None                              # set temporal case
        if temporal_idx is not None:
            tsel_idx = temporal_idx[idx[0], idx[2]]  # time step index of retained neighbors
        picked = flat_grid.read_values(node_idx, tsel_idx)

        # scatter each variable's valid values back to (n, K, N)
        values = {}
        for var, vals in picked.items():
            full = np.full(valid.shape, np.nan)
            full[idx] = vals
            values[var] = full

        # one neighbor in space and in time: back to one value per point, (n,)
        if valid.shape[1:] == (1, 1):
            values = {var: v[:, 0, 0] for var, v in values.items()}
            dist_km = dist_km[:, 0]
            time_delta = time_delta[:, 0]
            valid = valid[:, 0, 0]

        # distance and time gap are kept for every neighbor, valid tells which ones are matched
        return MatchupResult(values=values, distance_km=dist_km,
                             time_delta=time_delta, valid=valid)




class Interpolator(Method):
    """Interpolation based on Nearest-neighbor fit method

    Each point takes the values of the combination of closest nodes of the grid,
    and of the nearests times step if the grid has a time axis.
    Points beyond ``radius`` or ``time_gap`` are invalid.

    Parameters
    ----------
    radius: int, default=25
        Maximum distance to the nodes, in km (straight line on the sphere). Nodes beyond have
        no weight.
    time_gap: numpy.timedelta64, default=1 day
        Maximum gap to the time steps. Ignored for a grid without time axis.
    k_nearest: int, default=2
        Number of nodes weighted in space.
    n_frames: int, default=1
        Number of time steps weighted. Ignored for a grid without time axis.
    kernel: {'idw'}, default='idw'
        Weighting function of the distances and time gaps, see ``available_kernels``.
        'idw': inverse distance weighting.
    power: int, default=2
        Power of the distance in the kernel: the higher, the more the nearest nodes weigh.
    """
    optional_params = {"radius", "time_gap", "k_nearest", "n_frames", "kernel", "power"}

    def __init__(self, 
                 radius: int = 25,
                 time_gap: np.timedelta64 = np.timedelta64(1, "D"),
                 k_nearest : int = 2,
                 n_frames: int = 1,
                 kernel: str = "idw",
                 power: int = 2
                 ) -> None :
        self.radius = radius
        self.time_gap = time_gap
        self.k_nearest = k_nearest
        self.n_frames = n_frames
        self.kernel = kernel
        self.power = power


    def apply(self, dataset: xr.Dataset, points: PointSet) -> MatchupResult:
        """Match the points with a weighted combination of the nearest nodes and time steps

        Invalid points, with no node within ``radius`` or no time step within ``time_gap``,
        have NaN values in the result.

        Parameters
        ----------
        dataset: xarray.Dataset
            Grid with the ``lon`` and ``lat`` coordinates, and ``time`` if it has a time axis.
        points: :class:`.PointSet`

        Returns
        -------
        :class:`.MatchupResult`

        Raises
        ------
        ValueError
            If the grid has a time axis and the points have no time.
        """

        # starting by lonlat2xy on spatial grid
        # return FlatGrid object, flatten grid
        flat_grid = FlatGrid.from_grid(dataset)

        # length of points to process
        idx_count = len(points.lon)

        # k nearest nodes and n nearest time steps, sorted from the nearest
        (dist_km, spatial_idx, valid_spatial,
         time_delta, temporal_idx, valid_time) = self._neighbors(flat_grid, points,
                                                                 self.k_nearest, self.n_frames)

        # load kernel 
        kernel = available_kernels[self.kernel]

        # compute spatial weights
        # return (n, K) matrix
        # invalid nodes (beyond radius or missing) weigh 0
        spatial_weights = np.where(valid_spatial,
                                   kernel(dist_km, power=self.power, limit=self.radius), 0.0)

        # compute temporal weights 
        if flat_grid.time is None:
            temporal_weights = np.ones((idx_count, 1))      # no time axis
        else:
            # invalid time steps (beyond time_gap or missing) weigh 0
            temporal_weights = np.where(valid_time,
                                        kernel(time_delta, power=self.power,
                                               limit=self.time_gap_seconds), 0.0)
        
        # read EVERY points and let the equation select good points 
        values = {}
        for var in flat_grid.variables:

            # load variable into memory, one variable at a time
            var_picked = flat_grid.read_variable(var, spatial_idx, temporal_idx) # return (n, K, N) matrix
            if temporal_idx is None:
                var_picked = var_picked[:, :, None] # return (n, K, None) matrix
                
            # apply weights for each time step
            # (n, N), 1 value wighted per time frame
            var_weighted_sum = np.nansum(var_picked * spatial_weights[:, :, None], axis=1)
            # don't take nan into account # (n, N), 1 weight per time frame
            weights_sum = np.nansum(spatial_weights[:,:,None] * ~np.isnan(var_picked), axis=1)

            var_weighted_frames = np.full(var_weighted_sum.shape, np.nan, dtype=var_picked.dtype) # (n, N)
            np.divide(var_weighted_sum, weights_sum, 
                        out=var_weighted_frames, where=weights_sum > 0) # (n, N)

            # contract the frames : (n, N) -> (n,)
            var_weighted_sum_time = np.nansum(var_weighted_frames * temporal_weights, axis=1)
            weights_sum_time = np.nansum(temporal_weights * ~np.isnan(var_weighted_frames), axis=1)

            var_weighted_single = np.full(idx_count, np.nan, dtype=var_picked.dtype)

            np.divide(var_weighted_sum_time, weights_sum_time,
                      out=var_weighted_single, where=weights_sum_time > 0)
            
            values[var] = var_weighted_single

        valid = np.asarray(valid_spatial.any(axis=1) & valid_time.any(axis=1))          # (n,)

        # distance and time gap of the nearest node and time step (first ones, sorted),
        # kept for every point
        return MatchupResult(values=values,
                             distance_km=dist_km[:, 0],
                             time_delta=time_delta[:, 0],
                             valid=valid)

# key given to FloatMatcher.set_method(type=...) -> method class
available_methods: dict[str, type[Method]] = {
    "nearest": NearestNeighbor, "interp": Interpolator
}
"""dict: Method classes by name, used by :meth:`.FloatMatcher.set_method`

Available methods currently are:

- ``"nearest"``: :class:`.NearestNeighbor`
- ``"interp"``: :class:`.Interpolator`
"""
