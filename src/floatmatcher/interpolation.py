# interpolation.py / retourne un matchup interpolé aux localisations des profileurs argo.  

from typing import Any, cast

import xarray as xr
import numpy as np

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


class Interpolation(MatchupMethod):
    def __init__(self, method: str = "linear") -> None:
        self.method = method              # "linear", "nearest", "cubic"…

    def match(self, grid: GridSet, points: PointSet,
              constraints: Constraints) -> MatchupResult:
        ds = pad_periodic_lon(grid.dataset)                
        interp_coords = dict(
            lon=xr.DataArray(points.lon_in(grid.lon_range), dims="pts"),
            lat=xr.DataArray(points.lat, dims="pts"),
        )
        if grid.regime == "3D":          
            interp_coords["time"] = xr.DataArray(points.time, dims="pts")

        out = ds.interp(coords=interp_coords, method=cast(Any, self.method))
 
        values = {str(var): out[var].values for var in ds.data_vars}

        valid = ~np.isnan(next(iter(values.values())))

        return MatchupResult(values, 
                            distance_km=np.full(len(points.lon), np.nan), 
                            time_delta=np.full(len(points.lon), np.nan), 
                            valid=valid
                            )
        