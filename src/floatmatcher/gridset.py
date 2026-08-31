# gridset.py: validated wrapper around a normalized grid dataset

from dataclasses import dataclass, field
import xarray as xr
import numpy as np 

@dataclass
class GridSet:
    """Thin validated wrapper around a normalized grid dataset.

    Wraps a normalized ``xr.Dataset`` (coords ``lat``/``lon``, and ``time`` in
    the 3D case), validating it at construction and deriving its regime. The
    dataset is kept intact and accessible as ``grid.dataset``
    """

    dataset: xr.Dataset
    # convention the Product promises; None -> skip the check
    regime: str = field(init=False)     # "2D" or "3D", derived at construction

    def __post_init__(self) -> None:
        
        if "lon" not in self.dataset.coords or "lat" not in self.dataset.coords:
            raise ValueError(
                "The dataset given to GridSet object doesn't have lon or lat "
                "coordinates"
            )

        if len(self.dataset.data_vars)<1:
            raise ValueError("There is no variable in the dataset given to GridSet")

        # test of lat/lon unicity
        lon = self.dataset["lon"].values
        lat = self.dataset["lat"].values
        if len(np.unique(lon)) != len(lon):
            raise ValueError("grid: duplicated longitudes in array")
        if len(np.unique(lat)) != len(lat):
            raise ValueError("grid: duplicated latitudes in array")
        
        # select regime 3D/2D
        if "time" in self.dataset.coords : 
            self.regime = "3D"
        else:
            self.regime = "2D"

        # test time unicity if 3D regime
        if self.regime == "3D":
            times = self.dataset["time"].values
            if len(np.unique(times)) != len(times):
                raise ValueError("grid: duplicate timestamps (overlapping files?)")
