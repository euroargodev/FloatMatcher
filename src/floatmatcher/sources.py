# sources.py: where the product files come from : local/remote

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import copy
import os
import numpy as np
from .pointset import PointSet
from .resolver import FileResolver, PathTemplate, ExplicitFiles
from .utils import _indented_repr
import pandas as pd

class Source(ABC):
    """Where the files of a product come from

    A source gives the files a product needs for some points (``resolve``) and makes them
    available locally (``fetch``). Subclasses implement both.

    Parameters
    ----------
    path: str or list of str
        Directory or files.
    """
    name: str
    def __init__(self, 
                 path: str | Path | list[str],
                 product_info: dict[str, Any] | None = None
                 ) -> None:
            self.path = path
            self.product_info = product_info

    def __repr__(self) -> str:
        return _indented_repr(self)

    def build_request(self, points: PointSet,
                      variables: list[str] | None,
                      **options: list[str]) -> dict[str, Any]:
        raise ValueError("request can be build only if source == remote-cds")



    @abstractmethod
    def resolve(self, points: PointSet) -> list[str]:
        """Files needed for these points

        Parameters
        ----------
        points: :class:`.PointSet`

        Returns
        -------
        list of str
            Paths of the files.
        """

    @abstractmethod
    def fetch(self, points: PointSet, variables: list[str]) -> None:
        """Make the needed files available locally, downloading them if remote

        Parameters
        ----------
        points: :class:`.PointSet`
        """


class LocalSource(Source):
    """Files already on disk

    Given as a directory, a list of files, or a root and a path template. Nothing is downloaded.

    Parameters
    ----------
    path: str or list of str
        Directory or files. With a ``pattern``, the root of the tree.
    pattern: str, optional
        Path template under ``path``, built from the dates of the points with the fields
        ``{year}``, ``{month}``, ``{day}``, ``{hour}`` and ``{minute}``, for example
        ``"{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc"``. The neighboring days
        are taken too.

    Raises
    ------
    ValueError
        If ``path`` is empty, or if a ``pattern`` is given without a path or with a list of paths.
    """

    def __init__(self,
                 path: str | list[str] | Path,
                 pattern: str | None = None, 
                 product_info: dict[str, Any]| None = None) -> None:
        
        self.path = path
        self.pattern = pattern
        self.product_info = product_info
        self._resolver: FileResolver # declaration for mypy

        if self.pattern and not self.path:
            raise ValueError("pattern given without a root path")
        if not self.path:
            raise ValueError("no path or pattern provided for opening data")

        if self.pattern:
            # a pattern is substituted under ONE root, not a list of paths
            if not isinstance(self.path, (str, Path)):
                raise ValueError("a pattern needs a single root path, not a list")
            self._resolver = PathTemplate(self.path, self.pattern)
        else:
            self._resolver = ExplicitFiles(self.path)


    def resolve(self, points: PointSet) -> list[str]:
        """Files needed for these points

        A directory gives all its ``.nc`` files, recursively; a list gives its own files. With a
        ``pattern``, one file per date of the points and of the neighboring days. Missing files
        are skipped, with a warning for the dates of the points only (neighboring days are
        often absent at the edges of an archive).

        Parameters
        ----------
        points: :class:`.PointSet`

        Returns
        -------
        list of str
            Paths of the files.

        Raises
        ------
        FileNotFoundError
            If none of the files of a ``pattern`` exists.
        ValueError
            If the points have no time, with a ``pattern``.
        """
        return self._resolver.files_for(points)

    def fetch(self, points: PointSet, variables: list[str] )-> None:
        """Nothing to do, the files are already on disk"""
        pass
        

class CDSSource(Source):
    """
    TODO : ...
    """
    def __init__(self,
                 path: str | list[str] | Path,
                 product_info: dict[str, Any],
                 client: Any = None
                 ) -> None:
        
        self.path = path
        self.product_info: dict[str, Any] = product_info
        self._client = client     # cdsapi.Client, created on first use if not given

        # in remote case, resolver is always cache dir given in path
        self._resolver: FileResolver = ExplicitFiles(self.path)
        
        # request template of the product, completed from the points by build_request
        self.base_request: dict[str, Any] = copy.deepcopy(product_info["cds_request"])
        self.request: dict[str, Any] | None = None    # last request built

    @property
    def client(self) -> Any:
        """Client of the CDS API, ``cdsapi.Client()`` unless one was given"""
        if self._client is None:
            import cdsapi
            self._client = cdsapi.Client()
        return self._client
    

    def resolve(self, points: PointSet) -> list[str]:
        """ Files needed for these points """
        return self._resolver.files_for(points)

    def fetch(self, points: PointSet, variables:list[str])-> None:
        """ Get files with CDS API """
        # check cache path                   : if not a single path raises an error
        if not Path(self.path).exists():
            try:
                os.mkdir(self.path, 0o755)
            except:
                raise ValueError("Impossible to create cache dir")



    def _options_request(self, options: dict[str, list[str]]) -> dict[str, list[str]]:
        """Values of the product options for the request

        Parameters
        ----------
        options: dict
            Option -> values asked, a subset of ``product_info["cds_options"]``. An option not
            given takes all its values.

        Returns
        -------
        dict
            Option -> values, for every option of the product.

        Raises
        ------
        ValueError
            If an option is unknown, or its values are empty or not available.
        """
        available = self.product_info.get("cds_options", {})
        unknown = [o for o in options if o not in available]
        if unknown:
            raise ValueError(f"unknown options {unknown}, available: {list(available)}")

        values = {}
        for option, all_values in available.items():
            asked = options.get(option, all_values)
            if not asked:
                raise ValueError(f"{option}: empty list, available: {all_values}")
            wrong = [v for v in asked if v not in all_values]
            if wrong:
                raise ValueError(f"{option}: {wrong} not available, available: {all_values}")
            values[option] = list(asked)
        return values

    def build_request(self, points: PointSet,
                      variables: list[str] | None,
                      **options: list[str]) -> dict[str, Any]:
        """ 
        TODO 
        """
        # copy initial request
        request = copy.deepcopy(self.base_request)

        # Area covered by the float plus a margin and rounded outward on the ERA5 grid
        margin = self.product_info["grid_step"] *2        # degrees, at least the matchup radius (0.5° is about 55 km)
        grid_step = self.product_info["grid_step"]        # degrees, resolution of ERA5

        if points.lon.max() - points.lon.min() > 180:
            print(points.lon.max(), points.lon.min())
            raise ValueError("cross antemeridian : TODO : need to divide box in two")

        west = np.floor((points.lon.min() - margin) / grid_step) * grid_step
        east = np.ceil((points.lon.max() + margin) / grid_step) * grid_step
        south = max(np.floor((points.lat.min() - margin) / grid_step) * grid_step, -90)
        north = min(np.ceil((points.lat.max() + margin) / grid_step) * grid_step, 90)

        area = [north, west, south, east]

        request["area"] = area      # CDS expects this order

        # Variables: names in the files, translated to the names CDS expects
        cds_names = self.product_info["cds_variables"]
        if variables is None:
            raise ValueError(f"CDS needs a list of variables, available: {list(cds_names)}")
        unknown = [v for v in variables if v not in cds_names]
        if unknown:
            raise ValueError(f"unknown variables {unknown}, available: {list(cds_names)}")
        request["variable"] = [cds_names[v] for v in variables]

        # Options of the product (ERA5: time), all values if not given
        request.update(self._options_request(options))



        # time creation : 
        # par default, le time resolution = resolution de la grille min (pour ERA5= 1h)
        dates = np.unique(
                    np.asarray(points.time).astype(f"datetime64[{self.product_info["time_resolution"]}]")
                )
        print(dates)
        print("---")
        # ensuite il faut construire le pad autour pour avoir toutes les dates encadrantes.
        # TODO : Attention, granularité jour imposée par les champs year/month/day de la requête CDS 
        # TODO : il faut peut etre caler le time_margin sur time_gap ? 
        pad = self.product_info["time_margin"]
        if pad >= np.timedelta64(24, "h"):
            raise ValueError("time_margin must be less than 24 h")
        dates_padded = pd.to_datetime(np.unique(
            np.concatenate([dates - pad, dates, dates + pad]).astype("datetime64[D]"))
            )
        print(dates_padded)
        print(dates_padded.day.unique())


        # ensuite on extrait les différentes heures, jours et mois dont on a besoin 
        request["year"] = sorted([f"{d:04d}" for d in dates_padded.year.unique()])
        request["month"] = sorted([f"{d:02d}" for d in dates_padded.month.unique()])
        request["day"] = sorted([f"{d:02d}" for d in dates_padded.day.unique()])





        # test if request fit in 
        self.request = request    # kept for download()
        return request

    def eval_request(self, request):
        pass
available_sources: dict[str, type[Source]] = {
    "local": LocalSource, 
    "remote-cds": CDSSource
    }
"""dict: Source classes by name, used by :class:`.Product`"""
