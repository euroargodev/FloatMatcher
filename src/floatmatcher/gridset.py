# gridset.py: validated wrapper around a normalized grid dataset

from dataclasses import dataclass, field
import xarray as xr

@dataclass
class GridSet:
    """Thin validated wrapper around a normalized grid dataset.

    Wraps a normalized ``xr.Dataset`` (coords ``lat``/``lon``, and ``time`` in
    the 3D case), validating it at construction and deriving its regime. The
    dataset is kept intact and accessible as ``grid.dataset``; the wrapper does
    not transform it.
    """

    dataset: xr.Dataset
    regime: str = field(init=False)   # "2D" or "3D", derived at construction

    def __post_init__(self) -> None:

        if "lon" not in self.dataset.coords or "lat" not in self.dataset.coords:
            raise ValueError("The dataset given to GridSet object doesn't have lon or lat coordinates")

        if len(self.dataset.data_vars)<1:
            raise ValueError("There is no variable in the dataset given to GridSet")

        if "time" in self.dataset.coords : 
            self.regime = "3D"
        else:
            self.regime = "2D"

        # TODO: check grid regularity and monotonicity on lat/lon