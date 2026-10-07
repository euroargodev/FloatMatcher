import xarray as xr
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any
from dask.utils import SerializableLock
from pathlib import Path
from .sources import available_sources
from .utils import indented_repr

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
    name: str
    coord_map: dict[str, str] = {}
    src_available: list[str] = []

    def __init__(self, 
                 source: str, 
                 path: str, 
                 selected_variables: list[str] | None,
                 **source_params: Any
                ) -> None:
        if selected_variables is not None and (
                not isinstance(selected_variables, list) or not selected_variables):
            raise ValueError(
                f"selected_variables must be None (keep all) or a non-empty list of variable names,\
                \n not {selected_variables!r}"
            )
        self.source = source    # local / api
        self.selected_variables = selected_variables

        self._src_dataset = None
        self._regime: str | None = None
        self._available_variables = None

        if self.source not in self.src_available:
            raise ValueError(f"{type(self).__name__} supports {self.src_available}, not {source!r}")
      
        self._source = available_sources[source](path, **source_params)




    # allow to store path in ._source object but to access it from Product 
    # avoid duplication of path storage into Product and Source 
    @property
    def path(self) -> str | Path | list[str]:
        return self._source.path

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

    def _src_dataset_checker(self, dataset: xr.Dataset) -> None:
        if "lon" not in dataset.coords or "lat" not in dataset.coords:
            raise ValueError(
                "The dataset given to GridSet object doesn't have lon or lat "
                "coordinates"
            )

        if len(dataset.data_vars)<1:
            raise ValueError("There is no variable in the dataset given to GridSet")

        # test of lat/lon unicity
        lon = dataset["lon"].values
        lat = dataset["lat"].values
        if len(np.unique(lon)) != len(lon):
            raise ValueError("grid: duplicated longitudes in array")
        if len(np.unique(lat)) != len(lat):
            raise ValueError("grid: duplicated latitudes in array")

    def _id_regime(self, dataset: xr.Dataset) -> None:
        # select regime 3D/2D
        if "time" in dataset.coords : 
            self._regime = "3D"
            # test time unicity if 3D regime
            times = dataset["time"].values
            if len(np.unique(times)) != len(times):
                raise ValueError("grid: duplicate timestamps (overlapping files?)")
        else:
            self._regime = "2D"
    
    def __repr__(self) -> str:
        return indented_repr(self)

    @abstractmethod
    def normalize(self, ds_raw: xr.Dataset) -> xr.Dataset:
        ...


class ERA5Product(Product):
    name = "era5"
    coord_map = {"longitude": "lon", "latitude": "lat", "valid_time": "time"}
    src_available = ["local"]

    def normalize(self, ds_raw: xr.Dataset) -> xr.Dataset:
        ds = to_standard(ds_raw, self.coord_map)
        return ds
    
class LUTProduct(Product):
    name = "lut"
    coord_map = {"lon": "lon", "lat": "lat"}
    src_available = ["local"]

    def normalize(self, ds_raw: xr.Dataset) -> xr.Dataset:
        ds = to_standard(ds_raw, self.coord_map)
        return ds





# key given to FloatMatcher.set_product(type=...) -> Product class
available_products: dict[str, type[Product]] = {
    "era5": ERA5Product,
    "lut" : LUTProduct
}