from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .pointset import PointSet
from .exceptions import ProfileFormatError

# ─────────────────────────────────────────────────────────────
#  Private xarray extraction helpers
#  (the name-lookup logic lives HERE, never in PointSet)
# ─────────────────────────────────────────────────────────────

def _find_key(ds: xr.Dataset, *names: str) -> str:
    """Base level: return the FIRST of `names` present in ds (coord or variable).
    Raise ProfileFormatError if none match. The only place that knows the
    lookup rule."""
    for name in names:
        if name in ds.coords or name in ds.variables:
            return name
    raise ProfileFormatError(f"No dataset name matches these candidates: {names}")


def _get(ds: xr.Dataset, *names: str) -> xr.DataArray:
    """Return the DataArray (xarray object) — used when .dims is needed after."""
    return ds[_find_key(ds, *names)]


def _extract(ds: xr.Dataset, *names: str) -> NDArray[Any]:
    """Return the .values (raw ndarray) — the common case."""
    return _get(ds, *names).values

def _to_profiles(ds: xr.Dataset) -> xr.Dataset:
    """Collapse an argopy measurement layout (N_POINTS) to profiles (N_PROF)."""
    accessor = getattr(ds, "argo", None)
    if accessor is None or not hasattr(accessor, "point2profile"):
        raise ProfileFormatError(
            "from_argopy_float received an N_POINTS dataset but the argopy 'argo' "
            "accessor is unavailable. Import argopy first (it registers the "
            "accessor), or pass an already profile-shaped dataset (N_PROF)."
        )
    profiles: xr.Dataset = accessor.point2profile()
    return profiles

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
        # origin_index = df index ; origin_dim = "index".
        return PointSet(df[lon], df[lat], df[time], origin_index=df.index.to_numpy(),
            origin_dim=df.index.name or "index",)

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
        time_arr = _extract(ds, time, "JULD")  # <-- multi-name ONLY here (minimal option)

        # 3. origin_index only if point_dim is an actual coord of ds
        if point_dim in ds.coords:
            origin_index = ds[point_dim].values  
        else:
            origin_index = np.arange(ds.sizes[point_dim])

        return PointSet(lon=lon_arr, 
                        lat=lat_arr, 
                        time=time_arr, 
                        origin_index=origin_index, 
                        origin_dim=str(point_dim)
                        )

    # --- 4. argopy float: an xarray Dataset from DataFetcher().load().data ---
    @staticmethod
    def from_argopy_float(ds: xr.Dataset) -> PointSet:
        """argopy float Dataset -> PointSet, ONE position per profile.
 
        argopy returns a measurement-level layout (dim ``N_POINTS``: lon/lat/time
        repeat across each profile's depth levels). Colocalization wants one
        position per profile, so collapse to the profile layout (dim ``N_PROF``)
        when a measurement layout is given.
        """
        if "N_POINTS" in ds.dims:
            ds = _to_profiles(ds)                 # N_POINTS -> (N_PROF, N_LEVELS)
        return ProfileLoader.from_xrdataset(ds, lon="LONGITUDE", lat="LATITUDE",
                                            time="TIME")
 
    # --- 5. argopy index: a pandas DataFrame, one row per profile ---
    @staticmethod
    def from_argopy_index(index: pd.DataFrame) -> PointSet:
        """argopy index DataFrame -> PointSet.
 
        IndexFetcher().load().index is a DataFrame with longitude/latitude/date,
        one row per profile: exactly what from_dataframe consumes.
        """
        return ProfileLoader.from_dataframe(index, lon="longitude",
                                            lat="latitude", time="date")
