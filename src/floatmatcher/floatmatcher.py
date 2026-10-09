# FloatMatcher.py: the top-level coordinator (public API of the library).

from typing import Any

import numpy as np 
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .methods import Method, available_methods
# from .interpolation import Interpolation
from .pointset import PointSet
from .matchup_results import MatchupResult
from .product import Product, available_products

class FloatMatcher:
    """A `FloatMatcher` starts empty and is filled with the ``set_*`` methods:
    the points to match, the product to match them with, and the matching
    method. The function ``match()`` then runs the matchup.
    Results are stored in result object
    

    Attributes
    ----------
    points : PointSet or None
        Positions (lon, lat, time) to match.
    product : Product or None
        Gridded product the points are matched with.
    method : Method or None
        Matching method and its parameters.
    result : MatchupResult or None
        Result of the last ``match()`` call.

    Examples
    --------
    >>> fm = FloatMatcher()
    >>> fm.set_points_from_arrays(lon, lat, time)
    >>> fm.set_product("era5", "local", "/data/era5", ["sst"])
    >>> fm.set_method("nearest", radius=50, time_gap=np.timedelta64(3, "h"))
    >>> result = fm.match()
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
        self._result: MatchupResult | None = None



    def __repr__(self,) -> str:
        return (f"points:{self.points} \nproduct:{self.product} "
                f"\nvariables:{self.variables} \nmethod: {self.method}"
                f"\nresults: {self._result}")



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


    @property
    def result(self) -> MatchupResult:
        if self._result is None :
            raise AttributeError("no result yet, call match()")
        return self._result


    def match(self) -> None:
        if self.points is None:
            raise ValueError("match(): no points set, call set_points_from_arrays(...)")
        if self.product is None:
            raise ValueError("match(): no product set, call set_product(...)")
        if self.method is None:
            raise ValueError("match(): no method set, call set_method(...)")

        # Get files opened as a mfDataset in product.src_dataset
        self.product.open(points=self.points)

        self._result = self.method.apply(self.product.selected_dataset, self.points)



    def to_dataset(self) -> xr.Dataset:
        """Add the matched values to the Dataset the points come from.

        Values are added as ``<name>_coloc``. The source Dataset is not modified.

        Returns
        -------
        xarray.Dataset

        Raises
        ------
        ValueError
            No result yet, or points not loaded from a Dataset.
        TypeError
            Points loaded from a DataFrame, use ``to_dataframe()``.
        """
        if self._result is None or self.points is None:
            raise ValueError("no result yet or points are None, try call match()")
        
        return self._result._to_dataset(self.points)

    def to_dataframe(self) -> pd.DataFrame:
        """Add the matched values to the DataFrame the points come from.

        Values are added as ``<name>_coloc`` columns. The source DataFrame is not modified.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ValueError
            No result yet, or points not loaded from a DataFrame.
        TypeError
            Points loaded from a Dataset, use ``to_dataset()``.
        """
        if self._result is None or self.points is None:
            raise ValueError("no result yet or points are None, try call match()")
            
        return self._result._to_dataframe(self.points)
