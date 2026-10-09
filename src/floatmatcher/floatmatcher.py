# FloatMatcher.py: the top-level coordinator (public API of the library).

from typing import Any

import pandas as pd
import xarray as xr

from .methods import Method, available_methods
# from .interpolation import Interpolation
from .pointset import PointSet
from .matchup_results import MatchupResult
from .product import Product, available_products

class FloatMatcher:
    """Match positions with a gridded product.

    Fill it with the ``set_*`` methods (points, product, method), then call ``match()``
    and read the result with ``result`` or ``reinject()``.

    Attributes
    ----------
    points : PointSet or None
        Positions (lon, lat, time) to match.
    product : Product or None
        Product the points are matched with.
    method : Method or None
        Matching method and its parameters.
    result : MatchupResult
        Result of the last ``match()``. Raises if there is none yet.

    Examples
    --------
    >>> fm = FloatMatcher()
    >>> fm.set_points(lon, lat, time)
    >>> fm.set_product("era5", "local", "/data/era5", ["sst"])
    >>> fm.set_method("nearest", radius=50, time_gap=np.timedelta64(3, "h"))
    >>> fm.match()
    >>> fm.result
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



    def set_points(self, *data: Any, **names: str) -> None:
        """Set the points to match. The loader is chosen from the type of the input.

        Parameters
        ----------
        *data
            ``lon, lat[, time]`` arrays, a :class:`pandas.DataFrame`, a
            :class:`xarray.Dataset` or a :class:`PointSet`.
        **names : str
            ``lon``, ``lat``, ``time``: names of the columns or variables holding them.
            Only for a DataFrame or a Dataset.

        Raises
        ------
        TypeError
            Unsupported input, or names given with arrays or a PointSet.
        """
        if len(data) == 1 and not names and isinstance(data[0], PointSet):
            self.points = data[0]
        elif len(data) == 1 and isinstance(data[0], xr.Dataset):
            self.points = PointSet.from_xrdataset(data[0], **names)
        elif len(data) == 1 and isinstance(data[0], (pd.DataFrame, pd.Series)):
            self.points = PointSet.from_dataframe(data[0], **names)
        elif len(data) in (2, 3) and not names:
            self.points = PointSet.from_arrays(*data)
        else:
            raise TypeError(
                "set_points() expects lon, lat[, time] arrays, a DataFrame, a Dataset or a PointSet "
                "(lon=, lat=, time= are column or variable names, only for a DataFrame or a Dataset)"
            )



    def set_product(self,
                    type: str, 
                    source: str,
                    path: str, 
                    selected_variables: list[str] | None,
                    **src_params: Any
                    ) -> None:
        """Set the product to match with.

        Parameters
        ----------
        type : str
            ``"era5"`` or ``"lut"``.
        source : str
            Where the files come from. Only ``"local"`` for now.
        path : str or list of str
            Directory or files of the product. With a ``pattern``, the root of the tree.
        selected_variables : list of str or None
            Variables to keep. None keeps all.
        **src_params
            Parameters of the source. ``"local"``: ``pattern``, a path template like
            ``"{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc"``.

        Raises
        ------
        ValueError
            Unknown product, or source not available for it.
        TypeError
            Unknown source parameter.
        """
        if type not in available_products:
            raise ValueError(f"unknown product {type!r}, available: {list(available_products)}")
        
        self.product = available_products[type](source, 
                                                path, 
                                                selected_variables, 
                                                **src_params
                                                )
        

        
    def set_method(self, type: str, **params: Any) -> None:
        """Set the matching method.

        Parameters
        ----------
        type : str
            ``"nearest"``.
        **params
            Parameters of the method. ``"nearest"``: ``radius`` (km),
            ``time_gap`` (``numpy.timedelta64``), ``k_nearest``.

        Raises
        ------
        ValueError
            Unknown method.
        TypeError
            Unknown method parameter.
        """
        if type not in available_methods:
            raise ValueError(f"unknown method {type!r}, available: {list(available_methods)}")
        self.method = available_methods[type](**params)


    @property
    def result(self) -> MatchupResult:
        """Result of the last ``match()``.

        Raises
        ------
        AttributeError
            No result yet.
        """
        if self._result is None :
            raise AttributeError("no result yet, call match()")
        return self._result


    def match(self) -> None:
        """Run the matchup. Read the result with ``result``.

        Raises
        ------
        ValueError
            Points, product or method not set.
        """
        if self.points is None:
            raise ValueError("match(): no points set, call set_points(...)")
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

    def reinject(self) -> xr.Dataset | pd.DataFrame:
        """Add the matched values to the object the points come from.

        Same type as the input of ``set_points``: a Dataset gives a Dataset, a DataFrame
        gives a DataFrame. Values are added as ``<name>_coloc``. The source is not modified.

        Returns
        -------
        xarray.Dataset or pandas.DataFrame

        Raises
        ------
        ValueError
            No result yet, or points not loaded from a Dataset or a DataFrame.
        """
        if self._result is None or self.points is None:
            raise ValueError("no result yet or points are None, try call match()")
        origin = self.points.original_data
        if isinstance(origin, xr.Dataset):
            return self.to_dataset()
        if isinstance(origin, pd.DataFrame):
            return self.to_dataframe()
        raise ValueError(
            "Cannot reinject: these points come from arrays, there is no Dataset or DataFrame "
            "to add the values to. Use result."
        )
