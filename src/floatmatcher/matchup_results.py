# matchup_results.py: output object of matchup

import numpy as np
import pandas as pd
import xarray as xr

from dataclasses import dataclass
from numpy.typing import NDArray
from .pointset import PointSet


@dataclass
class MatchupResult:
    """Result of a matchup

    One value per point, in the order of the points. A point that is not matched (too far in
    space or in time) has NaN values, its distance and time gap are kept.

    Parameters
    ----------
    values: dict
        Matched values, one array per variable.
    distance_km: numpy.ndarray
        Distance to the nearest node, in km (straight line on the sphere, close to the
        great-circle distance at short range), for every point, matched or not.
    time_delta: numpy.ndarray
        Gap to the nearest time step, in seconds, for every point, matched or not. NaN for a
        grid without time axis.
    valid: numpy.ndarray of bool
        True for the matched points: a node within ``radius`` and a time step within
        ``time_gap``, whatever its value.
    """
    values:      dict[str, NDArray[np.float64]]
    distance_km: NDArray[np.float64]
    time_delta:  NDArray[np.float64]
    valid:       NDArray[np.bool_]

    def __repr__(self) -> str:
        return (f"\n    variables: {self.values.keys()} \
                \n    mean_dist:{np.round(np.nanmean(self.distance_km[self.valid]), 2)} (km)"
                f"\n    mean_time_delta:{np.round(np.nanmean(self.time_delta[self.valid]), 2)} (seconds) \
                \n    nb_match_found:{int(self.valid.sum())}")

    
    def _to_dataset(self, points: PointSet) -> xr.Dataset:
        """Reinject the matched values into the Dataset the points come from

        Values are added as ``<name>_coloc``, on the dimension of the points. The source is
        not modified.

        Parameters
        ----------
        points: :class:`.PointSet`

        Returns
        -------
        :class:`xarray.Dataset`

        Raises
        ------
        ValueError
            If the points have no origin.
        TypeError
            If the origin is not a Dataset.
        """
        ds = points.original_data
        dim = points.points_dim
        if ds is None or dim is None:
            raise ValueError(
                "Cannot reinject: these points have no origin dataset "
                "(they came from raw arrays). Use the MatchupResult directly."
            )
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"Cannot reinject into a Dataset: the origin is a {type(ds).__name__}. "
                "Use FloatMatcher.to_dataframe()."
            )

        out = ds.copy()
        for k, v in self.values.items():
            out[f"{k}_coloc"] = (dim, v)

        return out

    def _to_dataframe(self, points: PointSet) -> pd.DataFrame:
        """Reinject the matched values into the DataFrame the points come from

        Values are added as ``<name>_coloc`` columns. The source is not modified.

        Parameters
        ----------
        points: :class:`.PointSet`

        Returns
        -------
        :class:`pandas.DataFrame`

        Raises
        ------
        ValueError
            If the points have no origin.
        TypeError
            If the origin is not a DataFrame.
        """
        df = points.original_data
        if df is None:
            raise ValueError(
                "Cannot reinject: these points have no origin dataframe "
                "(they came from raw arrays). Use the MatchupResult directly."
            )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Cannot reinject into a DataFrame: the origin is a {type(df).__name__}. "
                "Use FloatMatcher.to_dataset()."
            )

        out = df.copy()
        for k, v in self.values.items():
            out[f"{k}_coloc"] = v

        return out
