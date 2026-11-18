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
        if self.valid.ndim == 3:
            # one valid neighbor per (point, node, time step): masks for each axis
            valid_nodes = self.valid.any(axis=2)          # (n, K): node used at least once
            valid_steps = self.valid.any(axis=1)          # (n, N): time step used at least once
            valid_points = self.valid.any(axis=(1, 2))    # (n,): point with a valid neighbor
        else:
            valid_nodes = valid_steps = valid_points = self.valid
        return (f"\n    variables: {self.values.keys()} \
                \n    mean_dist:{np.round(np.nanmean(self.distance_km[valid_nodes]), 2)} (km)"
                f"\n    mean_time_delta:{np.round(np.nanmean(self.time_delta[valid_steps]), 2)} (seconds) \
                \n    nb_match_found:{int(valid_points.sum())}")

    def neigh_to_cols(self, suffix: str = "_coloc") -> dict[str, NDArray[np.float64]]:
        """Matched values as one array (n,) per variable and neighbor

        A variable (n, K, N) gives ``<var>_coloc_s<k>_t<n>``, s being the rank of the node in
        distance and t the rank of the time step in time gap, both from the nearest (0). A
        variable already one value per point, (n,), gives ``<var>_coloc``. These are the names
        of the columns added by ``reinject()``.

        Parameters
        ----------
        suffix: str, default='_coloc'
            Added after the name of the variable, before the rank of the neighbor.

        Returns
        -------
        dict
            One array (n,) per variable and neighbor, ready to be DataFrame columns.

        Examples
        --------
        >>> pd.DataFrame(my_fm.result.neigh_to_cols())
        """
        out = {}
        for var, v in self.values.items():
            if v.ndim == 1:
                out[f"{var}{suffix}"] = v
                continue
            for k in range(v.shape[1]):
                for t in range(v.shape[2]):
                    out[f"{var}{suffix}_s{k}_t{t}"] = v[:, k, t]
        return out

    def _to_dataset(self, points: PointSet, neigh_to_vars: bool = False) -> xr.Dataset:
        """Reinject the matched values into the Dataset the points come from

        Values are added as ``<name>_coloc``, on the dimension of the points. With
        ``k_nearest`` or ``n_frames`` above 1, they also have the ``k_space`` and ``n_time``
        dimensions, or are split into one ``<name>_coloc_s<k>_t<n>`` variable per neighbor
        (see :meth:`neigh_to_cols`). The source is not modified.

        Parameters
        ----------
        points: :class:`.PointSet`
        neigh_to_vars: bool, default=False
            If set to True: one variable per neighbor, on the dimension of the points only, instead of the
            ``k_space`` and ``n_time`` dimensions. No effect with one neighbor.

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
        if neigh_to_vars:
            # one variable per neighbor, along the points
            for name, v in self.neigh_to_cols().items():
                out[name] = (dim, v)
        else:
            # one variable per matched variable, (n,) or (n, K, N)
            for k, v in self.values.items():
                dims = (dim,) if v.ndim == 1 else (dim, "k_space", "n_time")
                out[f"{k}_coloc"] = (dims, v)

        return out

    def _to_dataframe(self, points: PointSet) -> pd.DataFrame:
        """Reinject the matched values into the DataFrame the points come from

        Values are added as ``<name>_coloc`` columns, or one ``<name>_coloc_s<k>_t<n>`` column
        per neighbor with ``k_nearest`` or ``n_frames`` above 1 (see :meth:`neigh_to_cols`).
        The source is not modified.

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
        for name, v in self.neigh_to_cols().items():
            out[name] = v

        return out
