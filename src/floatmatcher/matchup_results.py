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
    points:      PointSet

    def to_dataset(self) -> xr.Dataset:
        """
        Reinject the colocalized values into the dataset
        The source Dataset travels inside the PointSet
        """
        ds = self.points.origin_ds
        dim = self.points.origin_dim
        if ds is None or dim is None:
            raise ValueError(
                "Cannot reinject: these points have no origin dataset "
                "(they came from raw arrays). Use the MatchupResult directly."
            )

        out = ds.copy()
        for k, v in self.values.items():
            out[f"{k}_coloc"] = (dim, v)

        return out
