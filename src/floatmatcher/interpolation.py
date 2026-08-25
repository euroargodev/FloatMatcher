# interpolation.py / retourne un matchup interpolé aux localisations des profileurs argo.  

from dataclasses import dataclass, field
import xarray as xr
import numpy as np

from .method import MatchupMethod
from .pointset import PointSet
from .gridset import GridSet
from .results import MatchupResult



class Interpolation(MatchupMethod):
    def __init__(self, method="linear"):
        self.method = method              # "linear", "nearest", "cubic"…

    def match(self, grid, points, constraints):
        ds = grid.dataset                 # grid est un GridSet ; on interpole le dataset
        interp_coords = dict(
            lon=xr.DataArray(points.lon_in(grid.lon_range), dims="pts"),
            lat=xr.DataArray(points.lat, dims="pts"),
        )
        if grid.regime == "3D":           # régime porté par le GridSet
            interp_coords["time"] = xr.DataArray(points.time, dims="pts")

        out = ds.interp(**interp_coords, method=self.method)
        values = {var: out[var].values for var in ds.data_vars}

        valid = ~np.isnan(next(iter(values.values())))

        return MatchupResult(values, 
                            distance_km=np.full(len(points.lon), np.nan), 
                            time_delta=np.full(len(points.lon), np.nan), 
                            valid=valid
                            )
        