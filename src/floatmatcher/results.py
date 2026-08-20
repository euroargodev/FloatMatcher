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
