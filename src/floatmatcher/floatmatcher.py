# FloatMatcher.py: the top-level coordinator (public API of the library).

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from .methods import Method, available_methods

from .pointset import PointSet
from .matchup_results import MatchupResult
from .product import Product, available_products
from .utils import _check_params

class FloatMatcher:
    """Positions matcher for gridded products

    Fill it with the ``set_*`` methods (points, product, method), then call ``match()`` and read
    the result with ``result``. Results can be integrated to original dataset with ``to_xarray()``
    or ``to_dataframe()``.

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
        fm.set_product("era5-hourly", "local", "/data/era5", ["sst"])
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
            :class:`.PointSet`.
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
                    store: str | Path | list[str | Path],
                    selected_variables: list[str] | None,
                    **src_params: Any
                    ) -> None:
        """Set the product to match with

        Parameters
        ----------
        type: {'era5-hourly', 'lut'}
            Product to use.
        source: {'local', 'remote-cds'}
            Where the files come from: already on disk (``"local"``), or downloaded from the
            Copernicus Climate Data Store (``"remote-cds"``).
        store: str, pathlib.Path or list of them
            Where the files of the product are on disk (not a zarr store).

            - When ``source="local"``: a file, a directory (all the ``.nc`` files found in it and in
              its sub-directories are opened), or a list of files and/or directories. With the
              ``pattern`` option, ``store`` must be a single directory, the root of the tree.
            - When ``source="remote-cds"``: the cache directory, a single directory where the files
              are downloaded, then opened from. Requests already downloaded there are not
              downloaded again.

            Warning: the time used by the library to open files is proportional to the
            number of opened files. It is much faster to open a restricted list of files
            than an entire archive. It is preferable to build the list before the matchup,
            however if it is not possible, using the ``pattern`` option on large archives
            helps to speed up the process and gives similar times to providing a list
            (see ``**src_params``).
        selected_variables: list of str or None
            Variables to match, with their names in the files (e.g. ``sst``, ``t2m``).

            - ``source="local"``: None keeps all the variables of the files.
            - ``source="remote-cds"``: a list is required to know what to download. The
              variables available depend on the product, see :class:`.ERA5Product`.
        **src_params
            Parameters depending on the ``source``.

            - ``source="local"``: ``pattern``, a path template under ``store``, built from the
              dates of the points with the fields ``{year}``, ``{month}``, ``{day}``, ``{hour}``
              and ``{minute}``, e.g. ``"{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc"``.
              Only the files needed by the points are opened, which spares the user from
              selecting them by hand: the daily file of each point, plus the day before and the
              day after (close to midnight, the nearest time step can be in the next or previous
              file). It speeds up the opening of the files, especially on a large archive.
              Works only with one file per day.
            - ``source="remote-cds"``: ``client`` (required, ``cdsapi.Client()``),
              ``clear_cache`` and the options of the product (ERA5: ``time``, e.g.
              ``["00:00", "12:00"]``), see :class:`.CDSSource`.

        Raises
        ------
        ValueError
            If the product is unknown, the source is not available for it, or
            ``selected_variables`` is neither None nor a non-empty list.
        TypeError
            If a source parameter is unknown, or a required one is missing.

        Examples
        --------
        .. code-block:: python

            # a local archive, one file per day, every file of the directory opened (slow)
            fm.set_product("era5-hourly", "local", "/data/era5", ["sst"])

            # the same archive with a pattern: only the files of the dates of the points (fast)
            fm.set_product("era5-hourly", "local", "/data/era5", ["sst"],
                           pattern="{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc")

            # an explicit list of files (fast)
            fm.set_product("era5-hourly", "local",
                           ["/data/era5_20180101.nc", "/data/era5_20180102.nc"], ["sst"])

            # downloaded from CDS into ./cache (slow the first time, then read from the cache)
            fm.set_product("era5-hourly", "remote-cds", "./cache", ["sst"], client=cdsapi.Client())
        """
        if type not in available_products:
            raise ValueError(f"unknown product {type!r}, available: {list(available_products)}")

        self.product = available_products[type](source,
                                                store,
                                                selected_variables,
                                                **src_params
                                                )
        

        
    def set_method(self, type: str, **params: Any) -> None:
        """Set the matching method

        Parameters
        ----------
        type: {'nearest', 'interp'}
            Method to use.
        **params
            Parameters of the method. ``"nearest"``: ``radius`` (km), ``time_gap``
            (numpy.timedelta64), ``k_nearest`` and ``n_frames`` (number of nearest nodes and
            time steps, 1 by default). 
            ``"interp"``: ``radius`` (km), ``time_gap``
            (numpy.timedelta64), ``k_nearest``, ``n_frames``, ``kernel`` and ``power``.

        Raises
        ------
        ValueError
            If the method is unknown.
        TypeError
            If a method parameter is unknown.
        """
        if type not in available_methods:
            raise ValueError(f"unknown method {type!r}, available: {list(available_methods)}")
        _check_params("method", type, params, available_methods[type].required_params,
                      available_methods[type]._optional_params())
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



    def to_xarray(self, neigh_to_vars: bool = False, in_place: bool = False) -> xr.Dataset:
        """Add the matched values to the Dataset the points come from

        Values are added as ``<name>_coloc``. The source Dataset is not modified.

        Only the values of the variables are added. Time gaps and distances are available in MtchupResult Only
        
        Parameters
        ----------
        neigh_to_vars: bool, default=False
            With ``k_nearest`` or ``n_frames`` above 1: one ``<name>_coloc_s<k>_t<n>`` variable
            per neighbor along the points, instead of the ``k_space`` and ``n_time``
            dimensions.
        in_place: bool, default=False
            If set to True, the original xarray dataset is enriched with matched variables instead of a copy. 

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

        if self.product is None or self.product.src_dataset is None:
            raise ValueError("product.src_dataset is None")

        return self._result._to_xarray(self.points, self.product.src_dataset,
                                       neigh_to_vars=neigh_to_vars, in_place=in_place)

    def to_dataframe(self) -> pd.DataFrame:
        """Add the matched values to the DataFrame the points come from

        Values are added as ``<name>_coloc`` columns. The source DataFrame is not modified.

        Only the values of the variables are added. Time gaps and distances are available in MtchupResult Only

        Returns
        -------
        :class:`pandas.DataFrame`

        Raises
        ------
        ValueError
            If there is no result yet, or the points come from arrays.
        TypeError
            If the points come from a Dataset, use ``to_xarray()``.
        """
        if self._result is None or self.points is None:
            raise ValueError("no result yet or points are None, try call match()")
            
        return self._result._to_dataframe(self.points)


    def build_request(self) -> list[dict[str, Any]]:
        """Build the download request of the product for the points

        Only for a product from ``"remote-cds"``: the request covers the area and the dates of
        the points, for the selected variables, split under the cost limit of CDS. Then call
        :meth:`download`. Optional: ``match()`` builds and downloads the request itself. The
        options of the product (ERA5: ``time``) are given to :meth:`set_product`.

        Returns
        -------
        list of dict
            Sub-requests to download.

        Raises
        ------
        ValueError
            If the points or the product are not set, if the source of the product has no
            request, or if the request can not be built.

        Examples
        --------
        .. code-block:: python

            fm.set_points(lon, lat, time)
            fm.set_product("era5-hourly", "remote-cds", "/cache", ["sst"], client=cdsapi.Client(),
                           time=["00:00", "12:00"])
            fm.build_request()
            fm.download()
        """
        if self.points is None:
            raise ValueError("build_request(): no points set, call set_points(...)")
        if self.product is None:
            raise ValueError("build_request(): no product set, call set_product(...)")

        res = self.product.build_request(self.points)

        return res

    def download(self,) -> None:
        """Download the request built by :meth:`build_request`

        Only for a product from ``"remote-cds"``. The files go to a folder of the cache
        directory, the product reads them from there. Nothing is downloaded if the request is
        already in the cache. Optional: ``match()`` builds and downloads the request itself.

        Raises
        ------
        ValueError
            If the product is not set, if its source has nothing to download, or if no request
            was built.
        requests.HTTPError
            If CDS refuses a sub-request.
        """
        if self.product is None:
            raise ValueError("download(): no product set, call set_product(...)")

        self.product.download()

    def show_register(self) -> pd.DataFrame:
        """Displays the requests stored in the register of requests in the cache directory

        Only for a product from ``"remote-cds"``, see :meth:`.CDSSource.show_register`.

        Returns
        -------
        :class:`pandas.DataFrame`
            One row per request, indexed by the uuid of its folder (``request``), with its
            state first, then its fields: ``variables``, ``area``, dates, ``time``,
            ``dataset``. Empty if nothing was downloaded in this cache yet.

        Raises
        ------
        ValueError
            If the product is not set, or if its source has no cache.
        """
        if self.product is None:
            raise ValueError("show_register(): no product set, call set_product(...)")

        return self.product.show_register()
