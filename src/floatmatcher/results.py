# results.py: output object of matchup

import numpy as np
import xarray as xr 

from dataclasses import dataclass
from numpy.typing import NDArray
from .pointset import PointSet


@dataclass
class MatchupResult:
    values:      dict[str, NDArray[np.float64]]   
    distance_km: NDArray[np.float64]        
    time_delta:  NDArray[np.float64]         
    valid:       NDArray[np.bool_]             

    def to_dataset(self, ds: xr.Dataset, points: PointSet) -> xr.Dataset:
        """Reinject the colocalized values into the original dataset."""
        dim = points.origin_dim
        if dim is None:
            raise ValueError(
                "Cannot reinject: these points have no origin dataset "
                "(they came from raw arrays). Use the MatchupResult directly."
            )
        if ds.sizes[dim] != len(self.valid):
            raise ValueError(
                f"Dataset size along '{dim}' ({ds.sizes[dim]}) does not match "
                f"the number of results ({len(self.valid)})."
            )

        out = ds.copy()
        for k, v in self.values.items():
            out[f"{k}_coloc"] = (dim, v)

        return out

    def update_best(self, other: "MatchupResult") -> "MatchupResult":
        """Replace current result if a new result presents better distances."""
        # `~self.valid` catches points where self has nothing yet: there
        # self.time_delta is NaN and the delta comparison alone would be False.
        improve = other.valid & (~self.valid | (other.time_delta < self.time_delta))
        self.distance_km = np.where(improve, other.distance_km, self.distance_km)
        self.time_delta = np.where(improve, other.time_delta, self.time_delta)
        for var in self.values:
            self.values[var] = np.where(improve, other.values[var], self.values[var])
        self.valid = self.valid | other.valid
        return self