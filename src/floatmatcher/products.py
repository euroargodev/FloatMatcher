import numpy as np
import xarray as xr 

# ─────────────────────────────────────────────────────────────
#  Private xarray extraction helpers
#  (the name-lookup logic lives HERE, never in PointSet)
# ─────────────────────────────────────────────────────────────


def rename_coords(ds, mapping):
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


def to_standard(ds, mapping):
    """Bring a raw grid dataset to the standard form expected by GridSet."""
    ds = rename_coords(ds, mapping)
    return ds


class ERA5Product:
    COORD_MAP = {"longitude": "lon", "latitude": "lat", "valid_time": "time"}
    LON_RANGE = "0-360"
    def normalize(self, raw):
        return to_standard(raw, self.COORD_MAP)    # la classe appelle la fonction

class LUTProduct:
    COORD_MAP = {"lon": "lon", "lat": "lat"}
    LON_RANGE = "-180-180"
    def normalize(self, raw):
        return to_standard(raw, self.COORD_MAP)