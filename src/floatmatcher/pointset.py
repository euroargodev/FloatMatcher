# pointset.py: validated container for the input points to colocalize

import warnings
from dataclasses import dataclass, field
import numpy as np
import xarray as xr
import pandas as pd


from numpy.typing import ArrayLike, NDArray
from .exceptions import ProfileFormatError
from .geo import lonlat_to_xyz
from .constants import TIME_UNIT
from .utils import extract, get


@dataclass(repr=False)
class PointSet:
    """Validated positions (lon, lat, time) to match

    Inputs are converted to NumPy arrays (float degrees, ``datetime64[ns]``) and checked for
    equal lengths. Heavy computation works on these arrays (``points.lon``), never on the
    object itself inside loops. Prefer :meth:`from_arrays`, :meth:`from_dataframe` or
    :meth:`from_xrdataset`.

    Parameters
    ----------
    lon, lat: array-like of float
        Degrees.
    time: array-like of datetime64, optional
        Time of each point.
    points_dim: str, optional
        Dimension holding the points in ``original_data``.
    original_data: pandas.DataFrame or xarray.Dataset, optional
        Object the points come from, kept to reinject the matched values.

    Raises
    ------
    ValueError
        If lon, lat and time do not have the same length.

    Examples
    --------
    .. code-block:: python

        points = PointSet.from_arrays(lon, lat, time)
    """

    lon: NDArray[np.float64]
    lat: NDArray[np.float64]
    time: NDArray[np.datetime64] | None = None
    points_dim: str | None = None       # source dimension name
    original_data: pd.DataFrame | xr.Dataset | None = None  # source dataset
    _xyz: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """conversion of lon/lat/time to canonical dtypes, then validation"""
        self.lon = np.asarray(self.lon, dtype=float)
        self.lat = np.asarray(self.lat, dtype=float)
        if self.time is not None:
            self.time = np.asarray(self.time, dtype=f"datetime64[{TIME_UNIT}]")
        self.check_lengths()

    def check_lengths(self) -> None:
        """Raise ValueError if lon, lat and time (when set) differ in length"""
        if self.time is not None:
            lengths = [len(self.lon), len(self.lat), len(self.time)]
        else:
            lengths = [len(self.lon), len(self.lat)]

        if len(np.unique(lengths)) != 1:
            raise ValueError("lon, lat, time must have the same length")

    def __repr__(self) -> str:
        if self.time is not None:
            return (f"\n    type: {type(self).__name__} \n    lon:{len(self.lon)} "
                    f"\n    lat:{len(self.lat)} \n    time:{len(self.time)}")
        else:
            return f"\n    lon:{len(self.lon)} \n    lat:{len(self.lat)}"



    @property
    def xyz(self) -> NDArray[np.float64]:
        """Cartesian coordinates (km) on the sphere, shape (n, 3), computed once and cached"""
        if self._xyz is None:
            self._xyz = lonlat_to_xyz(self.lon, self.lat)
        return self._xyz

    # --- 1. Raw arrays: simplest case ---
    @classmethod
    def from_arrays(cls,
                    lon: ArrayLike,
                    lat: ArrayLike,
                    time: ArrayLike | None = None) -> "PointSet":
        """Build from lon, lat[, time] arrays

        Parameters
        ----------
        lon, lat: array-like of float
            Degrees.
        time: array-like of datetime64, optional

        Returns
        -------
        :class:`PointSet`
        """
        # Nothing to extract, no origin_* (bare arrays). Converted in __post_init__.
        return cls(lon, lat, time)  # type: ignore[arg-type]


    # --- 2. pandas DataFrame ---
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame | pd.Series, lon: str = "longitude",
                        lat: str = "latitude", time: str | None = "date") -> "PointSet":
        """Build from the columns of a DataFrame, or from one row (Series)

        The DataFrame is kept in ``original_data`` (a row becomes a one-row DataFrame) and
        the name of its index in ``points_dim``.

        Parameters
        ----------
        df: pandas.DataFrame or pandas.Series
        lon: str, default="longitude"
        lat: str, default="latitude"
        time: str or None, default="date"
            Time column. None gives points without time, a column that does not exist too
            but with a warning.

        Returns
        -------
        :class:`PointSet`

        Raises
        ------
        ProfileFormatError
            If the lon or lat column does not exist.

        Warns
        -----
        UserWarning
            If the time column does not exist.
        """
        # Check the names first: a clear error for lon/lat, a warning for the optional time
        missing = [name for name in (lon, lat) if name not in df]
        if missing:
            raise ProfileFormatError(
                f"No column named {missing} in the DataFrame, available: {list(df.keys())}")
        if time is not None and time not in df:
            warnings.warn(f"No column named {time!r}: the points have no time. "
                          "Pass time=None to silence this warning.", stacklevel=2)
            time = None

        # Stable, named columns -> NO multi-name lookup here.
        # atleast_1d: a single row (pd.Series) gives scalars, turned into 1-element arrays
        lon_arr = np.atleast_1d(np.asarray(df[lon]))
        lat_arr = np.atleast_1d(np.asarray(df[lat]))
        time_arr = None
        if time is not None:
            time_arr = np.atleast_1d(np.asarray(df[time]))

        points_dim = str(df.index.name) if df.index.name is not None else "index"
        origin_df = df.to_frame().T.infer_objects() if isinstance(df, pd.Series) else df
        return cls(lon_arr, lat_arr, time_arr, original_data=origin_df, points_dim=points_dim)

    # --- 3. xarray Dataset ---
    @classmethod
    def from_xrdataset(cls, ds: xr.Dataset, lon: str = "LONGITUDE",
                        lat: str = "LATITUDE", time: str | None = "TIME") -> "PointSet":
        """Build from the variables of a Dataset (Argo names by default)

        The Dataset is kept in ``original_data`` and the dimension of the longitude variable
        in ``points_dim``.

        Parameters
        ----------
        ds: xarray.Dataset
        lon: str, default="LONGITUDE"
        lat: str, default="LATITUDE"
        time: str or None, default="TIME"
            Time variable, ``"JULD"`` when it is not found. None gives points without time,
            no such variable too but with a warning.

        Returns
        -------
        :class:`PointSet`

        Raises
        ------
        ProfileFormatError
            If the lon or lat variable is missing.

        Warns
        -----
        UserWarning
            If neither the time variable nor ``"JULD"`` exists.
        """
        # The point-dimension name is DISCOVERED (da_lon.dims[0]), never assumed.
        da_lon = get(ds, lon)
        point_dim = da_lon.dims[0]          # DISCOVERED, never assumed

        # 2. extract the three arrays as ndarrays
        lon_arr  = da_lon.values
        lat_arr  = extract(ds, lat)
        time_arr = None
        if time is not None:
            for name in [time, "JULD"]:
                if name in ds.coords or name in ds.variables:
                    time_arr = extract(ds, name)
                    break
            if time_arr is None:
                warnings.warn(f"No variable named {time!r} or 'JULD': the points have no time. "
                              "Pass time=None to silence this warning.", stacklevel=2)

        return PointSet(lon=lon_arr, 
                        lat=lat_arr, 
                        time=time_arr, 
                        points_dim=str(point_dim),
                        original_data=ds
                        )

    # --- TODO: 4. argopy ---
