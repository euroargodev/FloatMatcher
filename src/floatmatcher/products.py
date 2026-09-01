import xarray as xr
from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Sequence
from typing import Self
from dask.utils import SerializableLock


from .pointset import PointSet
from .resolver import FileResolver, ExplicitFiles, PathTemplate


_NETCDF_LOCK = SerializableLock()

# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────

def to_standard(ds: xr.Dataset, mapping: dict[str, str]) -> xr.Dataset:
    """
    Rename keys found in mapping. If key not found, skip and let it as it is.
    set variables lon/lat/time as coords if not already
    """
  
    renamable = set(ds.variables) | set(ds.dims)
    valid={}
    for src, dst in mapping.items() :
        if src in renamable:
            valid[src] = dst 
                     
    ds = ds.rename(valid)

    # transform data var into coordinate if not already
    to_promote = []
    for name in ("lon", "lat", "time"):
        if name in ds.data_vars:
            to_promote.append(name)

    ds = ds.set_coords(to_promote)
    return ds



class Product(ABC):
    COORD_MAP: dict[str, str] = {}

    def __init__(self, resolver: FileResolver | None = None) -> None:
        self.resolver = resolver
    
    @classmethod
    def from_local(cls, 
                   path: str | Path | Sequence[str] | None = None,
                   pattern: str | None = None) -> Self:
        resolver: FileResolver
        if pattern and not path:
            raise ValueError("pattern given without a root path")
        if not path:
            raise ValueError("no path or pattern provided for opening data")

        if pattern:
            # a pattern is substituted under ONE root, not a list of paths
            if not isinstance(path, (str, Path)):
                raise ValueError("a pattern needs a single root path, not a list")
            resolver = PathTemplate(path, pattern)
        else:
            resolver = ExplicitFiles(path)

        return cls(resolver)

    def files_for(self, points: PointSet | None = None) -> list[str]:
        if self.resolver is None:
            raise ValueError(
                "this product carries no source; build it with "
                f"{type(self).__name__}.from_local(...) before resolving files"
            )
        return self.resolver.files_for(points)

    def open_paths(self, paths: Sequence[str]) -> xr.Dataset:
        """ open several files as a xr mf_dataset with options
        // lock option fix made by AI to avoid segfault : open_mfdataset 
        returns dask arrays read by multiple threads, but netCDF4/HDF5 
        isn't thread-safe here: concurrent access to xarray's file cache 
        corrupted memory and killed the process. A single shared lock serialises
        those reads //
        """
        if not paths:
            raise ValueError("empty file list")
        return xr.open_mfdataset(
            paths,
            combine="by_coords",
            lock=_NETCDF_LOCK,
            engine="netcdf4",
            compat="override",
            coords="minimal",
        )
    
    @abstractmethod
    def normalize(self, ds_raw: xr.Dataset) -> xr.Dataset:
        ...


class ERA5Product(Product):
    COORD_MAP = {"longitude": "lon", "latitude": "lat", "valid_time": "time"}
    def normalize(self, raw: xr.Dataset) -> xr.Dataset:
        ds = to_standard(raw, self.COORD_MAP)
        return ds
    
class LUTProduct(Product):
    COORD_MAP = {"lon": "lon", "lat": "lat"}
    def normalize(self, raw: xr.Dataset) -> xr.Dataset:
        ds = to_standard(raw, self.COORD_MAP)
        return ds