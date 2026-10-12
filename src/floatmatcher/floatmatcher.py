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
    """Positions matcher for gridded products

    Fill it with the ``set_*`` methods (points, product, method), then call ``match()`` and read
    the result with ``result`` or ``reinject()``.

    Parameters
    ----------
    points: PointSet, optional
    product: Product, optional
    method: Method, optional
        Objects already built. Prefer the ``set_*`` methods.

    Attributes
    ----------
    points: PointSet or None
        Positions (lon, lat, time) to match.
    product: Product or None
        Product the points are matched with.
    method: Method or None
        Matching method and its parameters.
    result: MatchupResult
        Result of the last ``match()``. Raises if there is none yet.

    Examples
    --------
    .. code-block:: python

        fm = FloatMatcher()
        fm.set_points(lon, lat, time)
        fm.set_product("era5", "local", "/data/era5", ["sst"])
        fm.set_method("nearest", radius=50, time_gap=np.timedelta64(3, "h"))
        fm.match()
        fm.result
    """

    def __init__(self, 
                 points: PointSet | None = None, 
                 product: Product | None = None,
                 method: Method | None = None 
                 ) -> None:
        
        self.points = points
        self.product = product
        self.method = method
        self._result: MatchupResult | None = None



    def __repr__(self,) -> str:
        return (f"points:{self.points} \nproduct:{self.product} "
                f"\nmethod: {self.method}"
                f"\nresults: {self._result}")



    def set_points(self, *data: Any, **names: str) -> None:
        """Set the points to match

        The loader is chosen from the type of the input.

        Parameters
        ----------
        *data
            ``lon, lat[, time]`` arrays, a :class:`pandas.DataFrame`, a :class:`xarray.Dataset` or a
            :class:`PointSet`.
        **names: str
            ``lon``, ``lat``, ``time``: names of the columns or variables holding them. Only for a
            DataFrame or a Dataset.

        Raises
        ------
        TypeError
            If the input is not supported, or if names are given with arrays or a PointSet.
        ValueError
            If lon, lat and time do not have the same length.
        ProfileFormatError
            If the lon or lat column or variable is missing (DataFrame or Dataset).

        Warns
        -----
        UserWarning
            If the time column or variable is missing (DataFrame or Dataset).
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
        """Set the product to match with

        Parameters
        ----------
        type: {'era5', 'lut'}
            Product to use.
        source: {'local'}
            Where the files come from.
        path: str or list of str
            Directory or files of the product. With a ``pattern``, the root of the tree.
        selected_variables: list of str or None
            Variables to keep. None keeps all.
        **src_params
            Parameters of the source. ``"local"``: ``pattern``, a path template like
            ``"{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc"``.

        Raises
        ------
        ValueError
            If the product is unknown, the source is not available for it, or
            ``selected_variables`` is neither None nor a non-empty list.
        TypeError
            If a source parameter is unknown.
        """
        if type not in available_products:
            raise ValueError(f"unknown product {type!r}, available: {list(available_products)}")
        
        self.product = available_products[type](source, 
                                                path, 
                                                selected_variables, 
                                                **src_params
                                                )
        

        
    def set_method(self, type: str, **params: Any) -> None:
        """Set the matching method

        Parameters
        ----------
        type: {'nearest'}
            Method to use.
        **params
            Parameters of the method. ``"nearest"``: ``radius`` (km), ``time_gap``
            (numpy.timedelta64), ``k_nearest``.

        Raises
        ------
        ValueError
            If the method is unknown.
        TypeError
            If a method parameter is unknown.
        """
        if type not in available_methods:
            raise ValueError(f"unknown method {type!r}, available: {list(available_methods)}")
        self.method = available_methods[type](**params)


    @property
    def result(self) -> MatchupResult:
        """Result of the last ``match()``

        Raises
        ------
        AttributeError
            If there is no result yet.
        """
        if self._result is None :
            raise AttributeError("no result yet, call match()")
        return self._result


    def match(self) -> None:
        """Run the matchup

        Opens the product on the files needed by the points, then applies the method. Read the
        result with ``result``.

        Raises
        ------
        ValueError
            If the points, the product or the method are not set, if the product dataset fails
            a check, or if the points have no time and the grid has a time axis.
        FileNotFoundError
            If no file of the product is found for the points.
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
        """Add the matched values to the Dataset the points come from

        Values are added as ``<name>_coloc``. The source Dataset is not modified.

        Returns
        -------
        :class:`xarray.Dataset`

        Raises
        ------
        ValueError
            If there is no result yet, or the points come from arrays.
        TypeError
            If the points come from a DataFrame, use ``to_dataframe()``.
        """
        if self._result is None or self.points is None:
            raise ValueError("no result yet or points are None, try call match()")
        
        return self._result._to_dataset(self.points)

    def to_dataframe(self) -> pd.DataFrame:
        """Add the matched values to the DataFrame the points come from

        Values are added as ``<name>_coloc`` columns. The source DataFrame is not modified.

        Returns
        -------
        :class:`pandas.DataFrame`

        Raises
        ------
        ValueError
            If there is no result yet, or the points come from arrays.
        TypeError
            If the points come from a Dataset, use ``to_dataset()``.
        """
        if self._result is None or self.points is None:
            raise ValueError("no result yet or points are None, try call match()")
            
        return self._result._to_dataframe(self.points)

    def reinject(self) -> xr.Dataset | pd.DataFrame:
        """Add the matched values to the object the points come from

        Same type as the input of ``set_points``: a Dataset gives a Dataset, a DataFrame gives a
        DataFrame. Values are added as ``<name>_coloc``. The source is not modified.

        Returns
        -------
        :class:`xarray.Dataset` or :class:`pandas.DataFrame`

        Raises
        ------
        ValueError
            If there is no result yet, or the points come from arrays.
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
