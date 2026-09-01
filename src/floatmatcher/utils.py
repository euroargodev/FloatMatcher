# utils.py

# All general methods are grouped here to avoid "@staticmethod" 
# in classes and lightning the writings. Furthermore, reusable 
# in other files without whole class import

import xarray as xr


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
