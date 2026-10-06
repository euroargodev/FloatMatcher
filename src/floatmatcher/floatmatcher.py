# FloatMatcher.py: the top-level coordinator (public API of the library).

from collections.abc import Sequence
from typing import Any

import numpy as np 
from numpy.typing import NDArray

from .gridset import GridSet
from .methods import Method, available_methods
# from .interpolation import Interpolation
from .pointset import PointSet
from .matchup_results import MatchupResult
from .product import Product, ERA5Product, LUTProduct
from .utils import _select_variables

class FloatMatcher:
    """FloatMatcher gather all Points Pointset(), variables needed, setup Product(). 
    It prepares all elements to call the method in .match() function. 
    All constraints and parametrization lives in .match()
    """

    def __init__(self, 
                 points: PointSet | None = None, 
                 product: Product | None = None,
                 variables: str | list[str] | None = None,
                 method: Method | None = None 
                 ) -> None:
        
        self.points = points
        self.product = product
        self.variables = variables
        self.method = method
        self._files: list[str] | None = None


    def __repr__(self,) -> str:
        return f"points:{self.points} \nproduct:{self.product} \nvariables:{self.variables} \nmethod: {self.method}"



    def set_points_from_arrays(self, 
                               lon: NDArray[np.float64] ,   
                               lat: NDArray[np.float64],
                               time: NDArray[np.datetime64] | None = None):
        self.points = PointSet(lon, lat, time)



    def set_local_product(self,
                    type:ERA5Product|LUTProduct, 
                    subset_variables: list | str | None = None,
                    path: list | str | None = None
                    ):
        
        self.product = Product()

        
    def set_method(self, type: str, **params: Any) -> None:
        if type not in available_methods:
            raise ValueError(f"unknown method {type!r}, available: {list(available_methods)}")
        self.method = available_methods[type](**params)


    @property
    def files(self) -> list[str]:
        if self._files is None:
            # always give points just not used in case of ExplicitFiles resolver
            self._files = self.product.files_for(self.points)
        return self._files




    def match(self) -> MatchupResult:
        if self.points is None:
            raise ValueError("match(): no points set, call set_points_from_arrays(...)")
        if self.product is None:
            raise ValueError("match(): no product set, call set_local_product(...)")
        if self.method is None:
            raise ValueError("match(): no method set, call set_method(...)")
        return self.method.apply(self._open_lazy_grid(self.files), self.points)


    def _open_lazy_grid(self, paths: Sequence[str]) -> GridSet:
        """Open set of files -> normalized, variable selected and validated GridSet."""
        ds_raw = self.product.open_paths(paths)
        ds = _select_variables(self.product.normalize(ds_raw), self.variables)
        return GridSet(ds)
    
