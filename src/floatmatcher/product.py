import xarray as xr
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any
from dask.utils import SerializableLock
from pathlib import Path
from .sources import available_sources

from .utils import indented_repr, _select_variables
from .pointset  import PointSet


_NETCDF_LOCK = SerializableLock()

# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────

def to_standard(ds: xr.Dataset, mapping: dict[str, str]) -> xr.Dataset:
    """Rename the dataset names found in a mapping to standard names

    Names of the mapping that are not in the dataset are skipped. The ``lon``, ``lat`` and
    ``time`` data variables are set as coordinates.

    Parameters
    ----------
    ds: xarray.Dataset
    mapping: dict
        Names of the product -> standard names.

    Returns
    -------
    :class:`xarray.Dataset`
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
    """Gridded product the points are matched with

    A product knows where its files come from (its source), opens them as one dataset with
    standard names (lon, lat, time) and keeps it in ``src_dataset``. Subclasses give their
    ``name``, ``coord_map`` and ``src_available``, and implement ``normalize``.

    Parameters
    ----------
    source: str
        Where the files come from, one of ``src_available``.
    path: str or list of str
        Directory or files of the product.
    selected_variables: list of str or None
        Variables to keep. None keeps all.
    **source_params
        Parameters of the source. ``"local"``: ``pattern``.

    Attributes
    ----------
    name: str
        Key of the product in ``available_products``.
    coord_map: dict
        Names of the product -> standard names.
    src_available: list of str
        Sources the product can be read from.
    src_dataset: xarray.Dataset or None
        Complete dataset, set by ``open()``.

    Raises
    ------
    ValueError
        If ``source`` is not in ``src_available``, or ``selected_variables`` is neither None
        nor a non-empty list.
    TypeError
        If a source parameter is unknown.
    """
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

        self.src_dataset : xr.Dataset | None = None

        if self.source not in self.src_available:
            raise ValueError(f"{type(self).__name__} supports {self.src_available}, not {source!r}")
      
        self._source = available_sources[source](path, **source_params)

    # TODO : get essential product metadata 

    # allow to store path in ._source object but to access it from Product 
    # avoid duplication of path storage into Product and Source 
    @property
    def path(self) -> str | Path | list[str]:
        """Directory or files of the product, as given to the source"""
        return self._source.path

    # view of src_dataset restricted to selected_variables : no data copy,
    # src_dataset stays complete
    @property
    def selected_dataset(self) -> xr.Dataset:
        """Dataset restricted to ``selected_variables``, without copying the data

        Raises
        ------
        ValueError
            If the product is not opened.
        """
        if self.src_dataset is None:
            raise ValueError("product not opened, call open(points) first")
        return _select_variables(self.src_dataset, self.selected_variables)

    # property of available variables in dataset 
    @property
    def available_variables(self) -> list[str]:
        """Names of the variables of the opened dataset

        Raises
        ------
        ValueError
            If the product is not opened.
        """
        if self.src_dataset is None:
            raise ValueError("product not opened, call open(points) first")
        return [str(v) for v in self.src_dataset.data_vars]


    def _open_paths(self, paths: Sequence[str]) -> xr.Dataset:
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

    def _check_time_unicity(self, dataset: xr.Dataset) -> None:
        # a grid without time axis (2D) has nothing to check
        if "time" in dataset.coords:
            times = dataset["time"].values
            if len(np.unique(times)) != len(times):
                raise ValueError("grid: duplicate timestamps (overlapping files?)")

    def open(self, points: PointSet) -> None:
        """Open the files needed by the points as one dataset

        The dataset is normalized, checked (lon and lat, at least one variable, no duplicated
        longitude, latitude or timestamp) and kept in ``src_dataset``.

        Parameters
        ----------
        points: :class:`PointSet`
            Positions to match, they decide which files are needed.

        Raises
        ------
        ValueError
            If there is no file, or if the dataset fails a check.
        """
        files = self._source.resolve(points)
        ds = self._open_paths(files)
        ds = self.normalize(ds)
        self._src_dataset_checker(ds)
        self._check_time_unicity(ds)
        self.src_dataset = ds

    
    def __repr__(self) -> str:
        return indented_repr(self)

    @abstractmethod
    def normalize(self, ds_raw: xr.Dataset) -> xr.Dataset:
        """Rename the names of the raw dataset to standard names (lon, lat, time)

        Parameters
        ----------
        ds_raw: xarray.Dataset
            Dataset as opened from the files.

        Returns
        -------
        :class:`xarray.Dataset`
        """
        ...


class ERA5Product(Product):
    """ERA5 reanalysis of ECMWF

    The names ``longitude``, ``latitude`` and ``valid_time`` are renamed lon, lat and time.
    """
    name = "era5"
    coord_map = {"longitude": "lon", "latitude": "lat", "valid_time": "time"}
    src_available = ["local"]

    def normalize(self, ds_raw: xr.Dataset) -> xr.Dataset:
        ds = to_standard(ds_raw, self.coord_map)
        return ds
    
class LUTProduct(Product):
    """Look-up table (LUT) gridded product

    Its coordinates are already named lon and lat.
    """
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