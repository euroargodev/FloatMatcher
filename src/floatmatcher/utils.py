# utils.py

# All general methods are grouped here to avoid "@staticmethod" 
# in classes and lightning the writings. Furthermore, reusable 
# in other files without whole class import

import xarray as xr
from .exceptions import ProfileFormatError
from numpy.typing import NDArray
from typing import Any


def _select_variables(ds: xr.Dataset, variables: str | list[str] | None = None) -> xr.Dataset:
    """Keep only the requested variables. None -> keep all"""
    if variables is None:
        return ds
    
    wanted = [variables] if isinstance(variables, str) else list(variables)
    missing = [v for v in wanted if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"variables not found in the dataset: {missing}. "
            f"Available: {list(ds.data_vars)}."
        )
    return ds[wanted]


def find_key(ds: xr.Dataset, name: str) -> str:
    if name in ds.coords or name in ds.variables:
        return name
    raise ProfileFormatError(f"No dataset name matches: {name}")


def get(ds: xr.Dataset, name: str) -> xr.DataArray:
    """Return the DataArray object"""
    return ds[find_key(ds, name)]


def extract(ds: xr.Dataset, name: str) -> NDArray[Any]:
    """Return the .values (raw ndarray) — the common case."""
    return get(ds, name).values