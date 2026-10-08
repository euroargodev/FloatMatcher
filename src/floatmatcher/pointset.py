# pointset.py: validated container for the input points to colocalize

from dataclasses import dataclass, field
import numpy as np
import xarray as xr
import pandas as pd
from .exceptions import ProfileFormatError


from numpy.typing import ArrayLike, NDArray
from .geo import lonlat_to_xyz
from .constants import TIME_UNIT
from .utils import extract, get


@dataclass(repr=False)
class PointSet:
    """Validated wrapper around the (lon, lat, time) arrays to colocalize.

    The wrapper carries validation and clear names, but heavy computation
    works directly on the underlying NumPy arrays (``points.lon``), never on
    the object itself inside loops.

    Dataset travels along original_data
    """

    lon: NDArray[np.float64]
    lat: NDArray[np.float64]
    time: NDArray[np.datetime64] | None = None
    # TODO: introduce PointSetOrigin class (ds and df)
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
        """Cartesian 3D coordinates on the sphere, computed once and cached."""
        if self._xyz is None:
            self._xyz = lonlat_to_xyz(self.lon, self.lat)
        return self._xyz

    # --- 1. Raw arrays: simplest case ---
    @classmethod
    def from_arrays(cls,
                    lon: ArrayLike,
                    lat: ArrayLike,
                    time: ArrayLike | None = None) -> "PointSet":
        # Nothing to extract, no origin_* (bare arrays). Converted in __post_init__.
        return cls(lon, lat, time)  # type: ignore[arg-type]


    # --- 2. pandas DataFrame ---
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame | pd.Series, lon: str = "longitude",
                        lat: str = "latitude", time: str | None = "date") -> "PointSet":
        # Stable, named columns -> NO multi-name lookup here.
        # atleast_1d: a single row (pd.Series) gives scalars, turned into 1-element arrays
        lon_arr = np.atleast_1d(np.asarray(df[lon]))
        lat_arr = np.atleast_1d(np.asarray(df[lat]))
        time_arr = None
        if time is not None and time in df:
            time_arr = np.atleast_1d(np.asarray(df[time]))

        points_dim = str(df.index.name) if df.index.name is not None else "index"
        origin_df = df.to_frame().T.infer_objects() if isinstance(df, pd.Series) else df
        return cls(lon_arr, lat_arr, time_arr, original_data=origin_df, points_dim=points_dim)

    # --- 3. xarray Dataset ---
    @classmethod
    def from_xrdataset(cls, ds: xr.Dataset, lon: str = "LONGITUDE",
                        lat: str = "LATITUDE", time: str = "TIME") -> "PointSet":
        # The point-dimension name is DISCOVERED (da_lon.dims[0]), never assumed.
        da_lon = get(ds, lon)
        point_dim = da_lon.dims[0]          # DISCOVERED, never assumed

        # 2. extract the three arrays as ndarrays
        lon_arr  = da_lon.values
        lat_arr  = extract(ds, lat)
        time_arr = None
        candidates = [time, "JULD"]
        for name in candidates:
            if name in ds.coords or name in ds.variables:
                time_arr = extract(ds, name)
                break
        if time_arr is None:
            raise ProfileFormatError(
                f"No dataset name matches time candidates: {candidates}")


        return PointSet(lon=lon_arr, 
                        lat=lat_arr, 
                        time=time_arr, 
                        points_dim=str(point_dim),
                        original_data=ds
                        )

    # --- TODO: 4. argopy ---
