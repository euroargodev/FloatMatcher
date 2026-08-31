import xarray as xr
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Self
from dask.utils import SerializableLock


from .pointset import PointSet
from .resolver import FileResolver, ExplicitFiles, PathTemplate


_NETCDF_LOCK = SerializableLock()

# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────

def rename_coords(ds: xr.Dataset, mapping: dict[str, str]) -> xr.Dataset:
    """Rename source coordinate names to the standard ones (lat/lon[/time]).
    Rename only keys found in mapping. If key not found, skip and let it as it is.
    """
    # mapping is {source_name: standard_name}, e.g. {"longitude": "lon"}.
    # A renamable name can be a variable, a coordinate, or a bare dimension,
    # so we test membership against the union of variables and dims.
    renamable = set(ds.variables) | set(ds.dims)
    valid={}
    for src, dst in mapping.items() :
        if src in renamable:
            valid[src] = dst 
                     
    return ds.rename(valid)



class Product(ABC):
    COORD_MAP: dict[str, str] = {}

    def __init__(self, resolver: FileResolver | None = None) -> None:
        self.resolver = resolver
    
    @classmethod
    def from_local(cls, path: str | Path | Sequence[str] | None = None,
               pattern: str | None = None) -> Self:
        if path and pattern:
            resolver = PathTemplate(path, pattern)
        elif path and not pattern:
            resolver = ExplicitFiles(path)
        elif pattern and not path: 
            raise ValueError("pattern given without a root path")
        else:
            raise ValueError("no path or pattern provided for opening data")
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
        ds = rename_coords(raw, self.COORD_MAP)
        return ds
    
class LUTProduct(Product):
    COORD_MAP = {"lon": "lon", "lat": "lat"}
    def normalize(self, raw: xr.Dataset) -> xr.Dataset:
        ds = rename_coords(raw, self.COORD_MAP)
        return ds