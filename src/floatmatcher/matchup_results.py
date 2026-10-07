# results.py: output object of matchup

import numpy as np
import pandas as pd
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

    def _to_dataset(self, points: PointSet) -> xr.Dataset:
        """
        Reinject the colocalized values into the dataset
        The source Dataset travels inside the PointSet
        """
        ds = points.origin_ds
        dim = points.origin_dim
        if ds is None or dim is None:
            raise ValueError(
                "Cannot reinject: these points have no origin dataset "
                "(they came from raw arrays). Use the MatchupResult directly."
            )
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"Cannot reinject into a Dataset: the origin is a {type(ds).__name__}. "
                "Use to_dataframe()."
            )

        out = ds.copy()
        for k, v in self.values.items():
            out[f"{k}_coloc"] = (dim, v)

        return out

    def _to_dataframe(self, points: PointSet) -> pd.DataFrame:
        """
        Reinject the colocalized values into the dataframe
        The source DataFrame travels inside the PointSet
        """
        df = points.origin_ds
        if df is None:
            raise ValueError(
                "Cannot reinject: these points have no origin dataframe "
                "(they came from raw arrays). Use the MatchupResult directly."
            )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Cannot reinject into a DataFrame: the origin is a {type(df).__name__}. "
                "Use to_dataset()."
            )

        out = df.copy()
        for k, v in self.values.items():
            out[f"{k}_coloc"] = v

        return out
