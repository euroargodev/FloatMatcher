import numpy as np
import xarray as xr 

from .pointset import PointSet
from .exceptions import ProfileFormatError

# ─────────────────────────────────────────────────────────────
#  Private xarray extraction helpers
#  (the name-lookup logic lives HERE, never in PointSet)
# ─────────────────────────────────────────────────────────────

def _find_key(ds, *names):
    """Base level: return the FIRST of `names` present in ds (coord or variable).
    Raise ProfileFormatError if none match. The only place that knows the lookup rule."""
    for name in names:
        if name in ds.coords or name in ds.variables:
            return name
    raise ProfileFormatError(f"No dataset name matches these candidates: {names}")


def _get(ds, *names):
    """Return the DataArray (xarray object) — used when .dims is needed after."""
    return ds[_find_key(ds, *names)]


def _extract(ds, *names):
    """Return the .values (raw ndarray) — the common case."""
    return _get(ds, *names).values

# ─────────────────────────────────────────────────────────────
#  ProfileLoader: converges any point source into a PointSet.
#  Never validates itself: it extracts and delegates to PointSet.
# ─────────────────────────────────────────────────────────────

class ProfileLoader:

    # --- 1. Raw arrays: simplest case ---
    @staticmethod
    def from_arrays(lon, lat, time) -> PointSet:
        # Nothing to extract, no origin_* (bare arrays).
        return PointSet(lon, lat, time)

    # --- 2. pandas DataFrame ---
    @staticmethod
    def from_dataframe(df, lon="longitude", lat="latitude", time="date") -> PointSet:
        # Stable, named columns -> NO multi-name lookup here.
        # origin_index = df index ; origin_dim = "index".
        return PointSet(df[lon], df[lat], df[time], origin_index=df.index.to_numpy(),
            origin_dim=df.index.name or "index",)

    # --- 3. xarray Dataset ---
    @staticmethod
    def from_xrdataset(ds, lon="LONGITUDE", lat="LATITUDE", time="TIME") -> PointSet:
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
                        origin_dim=point_dim
                        )

    # TODO : 
    @staticmethod
    def from_argopy_float(obj) -> PointSet:
        pass

    # TODO : 
    @staticmethod
    def from_argopy_index(idx) -> PointSet:
        pass