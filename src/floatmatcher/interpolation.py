# interpolation.py / retourne un matchup interpolé aux localisations des profileurs argo.  

from typing import Any, cast

import xarray as xr
import numpy as np
from numpy.typing import NDArray

from .gridset import GridSet
from .method import Constraints, MatchupMethod
from .pointset import PointSet
from .results import MatchupResult

from .geo import is_global_lon
 
 
def pad_periodic_lon(ds: xr.Dataset) -> xr.Dataset:
    """Append first column to last on the right.
 
    On a global grid (only) the last node (e.g. 359.75) and the first (0) are physical
    neighbours, but the axis is discontinuous there, so xarray.interp returns
    NaN for a point falling between them --> copy the first node to position
    first+360, restoring continuity. Lazy-friendly (concat only).

    """
    lon = ds["lon"].values
    if not is_global_lon(lon):
        return ds
    edge = ds.isel(lon=[0]).assign_coords(lon=[float(lon[0]) + 360.0])
    padded: xr.Dataset = xr.concat([ds, edge], dim="lon")
    return padded


def _within_bounds(ds: xr.Dataset, regime: str, lon: NDArray[np.float64],
                   points: PointSet) -> NDArray[np.bool_]:
    """Points bracketed by the grid axes, i.e. interpolable."""
    lon_axis = ds["lon"].values
    lat_axis = ds["lat"].values
    inside: NDArray[np.bool_] = (
        (lon >= lon_axis.min()) & (lon <= lon_axis.max())
        & (points.lat >= lat_axis.min()) & (points.lat <= lat_axis.max())
    )
    if regime == "3D":
        time_axis = ds["time"].values
        inside &= (points.time >= time_axis.min()) & (points.time <= time_axis.max())
    return inside


class Interpolation(MatchupMethod):
    def __init__(self, method: str = "linear") -> None:
        self.method = method              # "linear", "nearest", "cubic"…

    def match(self, grid: GridSet, points: PointSet,
              constraints: Constraints) -> MatchupResult:
        ds = pad_periodic_lon(grid.dataset)
        lon = points.lon_in(grid.lon_range)
        interp_coords = dict(
            lon=xr.DataArray(lon, dims="pts"),
            lat=xr.DataArray(points.lat, dims="pts"),
        )
        if grid.regime == "3D":          
            interp_coords["time"] = xr.DataArray(points.time, dims="pts")

        out = ds.interp(coords=interp_coords, method=cast(Any, self.method))
 
        values = {str(var): out[var].values for var in ds.data_vars}

        valid = _within_bounds(ds, grid.regime, lon, points)

        return MatchupResult(values, 
                            distance_km=np.full(len(points.lon), np.nan), 
                            time_delta=np.full(len(points.lon), np.nan), 
                            valid=valid
                            )
        