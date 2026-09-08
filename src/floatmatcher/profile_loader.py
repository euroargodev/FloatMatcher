from typing import Any

import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .pointset import PointSet
from .exceptions import ProfileFormatError

# ─────────────────────────────────────────────────────────────
#  Private xarray extraction helpers
#  (the name-lookup logic lives HERE, never in PointSet)
# ─────────────────────────────────────────────────────────────

def _find_key(ds: xr.Dataset, name: str) -> str:
    if name in ds.coords or name in ds.variables:
        return name
    raise ProfileFormatError(f"No dataset name matches: {name}")


def _get(ds: xr.Dataset, name: str) -> xr.DataArray:
    """Return the DataArray object"""
    return ds[_find_key(ds, name)]


def _extract(ds: xr.Dataset, name: str) -> NDArray[Any]:
    """Return the .values (raw ndarray) — the common case."""
    return _get(ds, name).values


# ─────────────────────────────────────────────────────────────
#  ProfileLoader: converges any point source into a PointSet.
#  Never validates itself: it extracts and delegates to PointSet.
# ─────────────────────────────────────────────────────────────

class ProfileLoader:

    # --- 1. Raw arrays: simplest case ---
    @staticmethod
    def from_arrays(lon: Any, lat: Any, time: Any) -> PointSet:
        # Nothing to extract, no origin_* (bare arrays).
        return PointSet(lon, lat, time)

    # --- 2. pandas DataFrame ---
    @staticmethod
    def from_dataframe(df: pd.DataFrame, lon: str = "longitude",
                       lat: str = "latitude", time: str = "date") -> PointSet:
        # Stable, named columns -> NO multi-name lookup here.
        return PointSet(df[lon], df[lat], df[time],
                        origin_dim=df.index.name or "index")

    # --- 3. xarray Dataset ---
    @staticmethod
    def from_xrdataset(ds: xr.Dataset, lon: str = "LONGITUDE",
                       lat: str = "LATITUDE", time: str = "TIME") -> PointSet:
        # The point-dimension name is DISCOVERED (da_lon.dims[0]), never assumed.
        da_lon = _get(ds, lon)
        point_dim = da_lon.dims[0]          # DISCOVERED, never assumed

        # 2. extract the three arrays as ndarrays
        lon_arr  = da_lon.values
        lat_arr  = _extract(ds, lat)
        time_arr = None
        candidates = [time, "JULD"]
        for name in candidates:
            if name in ds.coords or name in ds.variables:
                time_arr = _extract(ds, name)
                break
        if time_arr is None:
            raise ProfileFormatError(
                f"No dataset name matches time candidates: {candidates}")


        return PointSet(lon=lon_arr, 
                        lat=lat_arr, 
                        time=time_arr, 
                        origin_dim=str(point_dim),
                        origin_ds=ds
                        )

    # # --- 4. argopy float: an xarray Dataset from DataFetcher().load().data ---
    # @staticmethod
    # def from_argopy_float(ds: xr.Dataset) -> PointSet:
    #     """argopy float Dataset -> PointSet, ONE position per profile.
 
    #     argopy returns a measurement-level layout (dim ``N_POINTS``: lon/lat/time
    #     repeat across each profile's depth levels). Colocalization wants one
    #     position per profile, so collapse to the profile layout (dim ``N_PROF``)
    #     when a measurement layout is given.
    #     """
    #     if "N_POINTS" in ds.dims:
    #         ds = _to_profiles(ds)                 # N_POINTS -> (N_PROF, N_LEVELS)
    #     return ProfileLoader.from_xrdataset(ds, lon="LONGITUDE", lat="LATITUDE",
    #                                         time="TIME")
 
    # # --- 5. argopy index: a pandas DataFrame, one row per profile ---
    # @staticmethod
    # def from_argopy_index(index: pd.DataFrame) -> PointSet:
    #     """argopy index DataFrame -> PointSet.
 
    #     IndexFetcher().load().index is a DataFrame with longitude/latitude/date,
    #     one row per profile: exactly what from_dataframe consumes.
    #     """
    #     return ProfileLoader.from_dataframe(index, lon="longitude",
    #                                         lat="latitude", time="date")
