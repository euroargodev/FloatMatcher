# gridset.py: validated wrapper around a normalized grid dataset

from dataclasses import dataclass, field
import xarray as xr
from .geo import detect_lon_range, is_monotonic, is_strictly_increasing

@dataclass
class GridSet:
    """Thin validated wrapper around a normalized grid dataset.

    Wraps a normalized ``xr.Dataset`` (coords ``lat``/``lon``, and ``time`` in
    the 3D case), validating it at construction and deriving its regime. The
    dataset is kept intact and accessible as ``grid.dataset``
    """

    dataset: xr.Dataset
    declared_lon_range: str | None = None   # convention the Product promises; None -> skip check
    regime: str = field(init=False)         # "2D" or "3D", derived at construction
    lon_range: str = field(init=False)      # "-180-180" or "0-360", derived from lon values

    def __post_init__(self) -> None:
        
        if "lon" not in self.dataset.coords or "lat" not in self.dataset.coords:
            raise ValueError("The dataset given to GridSet object doesn't have lon or lat coordinates")

        if len(self.dataset.data_vars)<1:
            raise ValueError("There is no variable in the dataset given to GridSet")

        # test of monotonic lat/lon 
        if not is_monotonic(self.dataset["lon"].values):
            raise ValueError("grid: 'lon' axis is not monotonic")
        if not is_monotonic(self.dataset["lat"].values):
            raise ValueError("grid: 'lat' axis is not monotonic")

        # select regime 3D/2D
        if "time" in self.dataset.coords : 
            self.regime = "3D"
        else:
            self.regime = "2D"

        # validates the lon_range values
        self.lon_range = detect_lon_range(self.dataset["lon"].values)
        if self.declared_lon_range is not None and self.declared_lon_range != self.lon_range:
            raise ValueError(
                f"grid: data longitude convention {self.lon_range} does not match "
                f"the declared convention {self.declared_lon_range}"
            )

        # test time monotony if 3D regime
        if self.regime == "3D" and not is_strictly_increasing(self.dataset["time"].values):
            raise ValueError("grid: 'time' axis must be strictly increasing")
