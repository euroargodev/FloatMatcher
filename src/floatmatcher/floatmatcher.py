# FloatMatcher.py: the top-level coordinator (public API of the library).

from typing import Any

import numpy as np 
from numpy.typing import NDArray

from .methods import Method, available_methods
# from .interpolation import Interpolation
from .pointset import PointSet
from .matchup_results import MatchupResult
from .product import Product, available_products

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
        self.result: MatchupResult | None = None


    def __repr__(self,) -> str:
        return (f"points:{self.points} \nproduct:{self.product} "
                f"\nvariables:{self.variables} \nmethod: {self.method}"
                f"\nresults: {self.result}")



    def set_points_from_arrays(self, 
                               lon: NDArray[np.float64],   
                               lat: NDArray[np.float64],
                               time: NDArray[np.datetime64] | None = None) -> None:
        self.points = PointSet(lon, lat, time)



    def set_product(self,
                    type: str, 
                    source: str,
                    path: str, 
                    selected_variables: list[str] | None,
                    **src_params: Any
                    ) -> None:

        if type not in available_products:
            raise ValueError(f"unknown product {type!r}, available: {list(available_products)}")
        
        self.product = available_products[type](source, 
                                                path, 
                                                selected_variables, 
                                                **src_params
                                                )
        

        
    def set_method(self, type: str, **params: Any) -> None:
        if type not in available_methods:
            raise ValueError(f"unknown method {type!r}, available: {list(available_methods)}")
        self.method = available_methods[type](**params)



    def match(self) -> MatchupResult:
        if self.points is None:
            raise ValueError("match(): no points set, call set_points_from_arrays(...)")
        if self.product is None:
            raise ValueError("match(): no product set, call set_local_product(...)")
        if self.method is None:
            raise ValueError("match(): no method set, call set_method(...)")

        # Get files opened as a mfDataset in product.src_dataset
        self.product.open(points=self.points)

        self.result = self.method.apply(self.product, self.points)
        return self.result


    
