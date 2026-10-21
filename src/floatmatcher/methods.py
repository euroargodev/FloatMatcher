# methods.py: matchup methods (contract and registry).
#
# The grid is opened ONCE by the product and handed to apply(): the spatial and temporal
# lookups run on the whole grid in a single pass (no temporal packets), and the values are
# read lazily, only at the valid points.

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from .flatgrid import FlatGrid
from .matchup_results import MatchupResult
from .pointset import PointSet
from .utils import _indented_repr
from .weights import available_kernels

class Method(ABC):
    """Prototype of the matchup methods

    A method holds its parameters and matches points with a grid in ``apply``. It knows
    nothing of the product or of :class:`.FloatMatcher`: it receives the opened grid and the
    points. Subclasses implement ``apply`` and are registered in ``available_methods``.
    """

    time_gap: np.timedelta64

    def __repr__(self) -> str:
        return _indented_repr(self)

    @property
    def time_gap_seconds(self) -> float:
        """``time_gap`` in seconds"""
        return float(self.time_gap / np.timedelta64(1, "s"))
    
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
    k_nearest: int, default=1
        Number of neighbors looked up. Only 1 works for now.
    """

    def __init__(self, radius: int = 25,
                 time_gap: np.timedelta64 = np.timedelta64(1, "D"),
                 k_nearest : int = 1) -> None :
        self.radius = radius
        self.time_gap = time_gap
        self.k_nearest = k_nearest

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

        # starting Nearest method : apply kdtree on spatial
        dist_km, spatial_idx = flat_grid.spatial_nearest(points, k=self.k_nearest)
        valid_spatial = dist_km <= self.radius

        idx_count = len(points.lon)
        if flat_grid.time is not None:
            time_delta, temporal_idx = flat_grid.temporal_nearest(points, k=self.k_nearest)
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
                                time_delta=time_delta_out, valid=valid)




class Interpolator(Method):
    """Interpolation based on Nearest-neighbor fit method

    Each point takes the values of the combination of closest nodes of the grid,
    and of the nearests times step if the grid has a time axis.
    Points beyond ``radius`` or ``time_gap`` are invalid.

    Parameters
    ----------
    radius: int, default=25
        Maximum distance to the node, in km (straight line on the sphere).
    time_gap: numpy.timedelta64, default=1 day
        Maximum gap to the time step. Ignored for a grid without time axis.
    k_nearest: int, default=1
        Number of neighbors looked up. Only 1 works for now.
    """

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
        """Match the points with the nearest node of the grid"""

        # starting by lonlat2xy on spatial grid
        # return FlatGrid object, flatten grid
        flat_grid = FlatGrid.from_grid(dataset)

        # apply kdtree on spatial : find k nearest distances and indexes of nearest points
        dist_km, spatial_idx = flat_grid.spatial_nearest(points, k=self.k_nearest)
        valid_spatial = dist_km <= self.radius
        # reshape to fit (n,) --> (n,K) in 1D:
        spatial_idx = spatial_idx.reshape(len(points.lon), -1) 
        dist_km = dist_km.reshape(len(points.lon), -1) 
        valid_spatial = valid_spatial.reshape(len(points.lon), -1) 

        # length of points to process
        idx_count = len(points.lon)

        # apply kdtree on time : find k nearest "time distances"
        if flat_grid.time is not None:
            time_delta, temporal_idx = flat_grid.temporal_nearest(points, k=self.n_frames)
            # Reshape also
            temporal_idx = temporal_idx.reshape(len(points.lon), -1) 
            time_delta = time_delta.reshape(len(points.lon), -1) 
            valid_time = time_delta <= self.time_gap_seconds
        else:
            temporal_idx = None
            time_delta = np.full((idx_count, 1), np.nan)            
            valid_time = np.ones((idx_count, 1), dtype=bool)

        # load kernel 
        kernel = available_kernels[self.kernel]

        # compute spatial weights
        # return (n, K) matrix
        spatial_weights = kernel(dist_km, power=self.power, limit=self.radius) * valid_spatial

        # compute temporal weights 
        if flat_grid.time is None:
            temporal_weights = np.ones((idx_count, 1))      # no time axis
        else:
            temporal_weights = kernel(time_delta, power=self.power,
                                      limit=self.time_gap_seconds) * valid_time
        
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

        return MatchupResult(values=values,
                             distance_km=np.where(valid, dist_km.min(axis=1), np.nan),
                             time_delta=np.where(valid, time_delta.min(axis=1), np.nan),
                             valid=valid)

# key given to FloatMatcher.set_method(type=...) -> method class
available_methods: dict[str, type[Method]] = {
    "nearest": NearestNeighbor, "interp": Interpolator
}
"""dict: Method classes by name, used by :meth:`.FloatMatcher.set_method`"""
