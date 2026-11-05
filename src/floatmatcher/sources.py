# sources.py: where the product files come from : local/remote

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import copy
import os
import json
from pprint import pprint
import numpy as np
from .pointset import PointSet
from .resolver import FileResolver, PathTemplate, ExplicitFiles
from .utils import _indented_repr
import pandas as pd
import zipfile
import shutil
import uuid

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
            self.original_request = None

    def __repr__(self) -> str:
        return _indented_repr(self)

    def build_request(self, points: PointSet,
                      variables: list[str] | None,
                      **options: list[str]) -> list[dict[str, Any]]:
        raise ValueError("request can be build only if source == remote-cds")

    def download(self, **options) :
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
                 client: Any,
                 clear_cache: bool
                 ) -> None:
        
        self.path = path
        self.product_info = product_info
        self.client = client      # cdsapi.Client, or mocking.FakeCDSClient to develop without network
        self.clear_cache = clear_cache


        # in remote case, resolver is the cache dir given in path, until download()
        # points it to the folder of the request
        self._resolver: FileResolver = ExplicitFiles(self.path)
        
        # request template of the product, completed from the points by build_request
        self.base_request: dict[str, Any] = copy.deepcopy(product_info["cds_request"])
        self.request: list[dict[str, Any]] | None = None    # last request built


    def resolve(self, points: PointSet) -> list[str]:
        """ Files needed for these points """
        return self._resolver.files_for(points)

    def fetch(self, points: PointSet, variables:list[str])-> None:
        """ fetch is buid request + download in order to 
        Get files with CDS API 
        """
        # verify path
        Path(self.path).mkdir(parents=True, exist_ok=True)

        # build requests
        self.build_request(points, variables)

        # download
        self.download()


        # update state ? 



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

    def _build_request(self, points: PointSet,
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
        # ensuite il faut construire le pad autour pour avoir toutes les dates encadrantes.
        # TODO : Attention, granularité jour imposée par les champs year/month/day de la requête CDS 
        # TODO : il faut peut etre caler le time_margin sur time_gap ? 
        pad = self.product_info["time_margin"]
        if pad >= np.timedelta64(24, "h"):
            raise ValueError("time_margin must be less than 24 h")
        dates_padded = pd.to_datetime(np.unique(
            np.concatenate([dates - pad, dates, dates + pad]).astype("datetime64[D]"))
            )


        # ensuite on extrait les différentes heures, jours et mois dont on a besoin 
        request["year"] = sorted([f"{d:04d}" for d in dates_padded.year.unique()])
        request["month"] = sorted([f"{d:02d}" for d in dates_padded.month.unique()])
        request["day"] = sorted([f"{d:02d}" for d in dates_padded.day.unique()])

        # test if request fit in 
        return request

    def eval_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Cost of a request, as estimated by CDS, without sending it

        Parameters
        ----------
        request: dict   

        Returns
        -------
        dict
            ``cost`` and ``limit`` of the request.
        """
        dataset_name = self.product_info["cds_dataset"]
        costs = self.client.client.estimate_costs(dataset_name, request)

        return costs

    def build_request(self, points: PointSet,
                      variables: list[str] | None,
                      **options: list[str]) -> list[dict[str, Any]]:
        """ 
        TODO 
        """
        # commence par build la premiere request,
        self.original_request = self._build_request(points, variables, **options)

        
        # eval cost of original request
        req_costs = self.eval_request(self.original_request)

        # subdivide request if necessary,
        # if req_costs["cost"] >= req_costs["limit"]:
        
        if req_costs["cost"] >= 200:
            requests = []
            for y in self.original_request["year"]:
                y_req = copy.deepcopy(self.original_request)
                y_req["year"] = [y]
                costs = self.eval_request(y_req)
                if costs["cost"] < costs["limit"]:
                    requests.append(y_req)
                    continue
                # trop gros : un mois à la fois
                for m in y_req["month"]:
                    m_req = copy.deepcopy(y_req)
                    m_req["month"] = [m]
                    costs = self.eval_request(m_req)
                    if not costs["cost"] < costs["limit"]:
                        raise ValueError("a single month exceeds limit, consider reduce area or variables or hours" )        # même un mois ne passe pas
                    requests.append(m_req)
            self.request = requests
            return self.request
        else:
            self.request = [self.original_request]
            return self.request


    @staticmethod
    def _is_included(request_info: dict[str, Any], registered: dict[str, Any]) -> bool:
        """Whether a request is covered by a request of the register

        Parameters
        ----------
        request_info: dict
            Request to download, as written in the register.
        registered: dict
            Request of the register.

        Returns
        -------
        bool
            True if same dataset, area inside the registered one, and every variable, date,
            hour and pressure level among the registered ones.
        """
        if request_info["dataset"] != registered["dataset"]:
            return False

        # area: [north, west, south, east]
        north, west, south, east = request_info["area"]
        r_north, r_west, r_south, r_east = registered["area"]
        if not (north <= r_north and west >= r_west and south >= r_south and east <= r_east):
            return False

        # CDS takes every combination of these lists, so each list included is enough
        for key in ["variables", "year", "month", "day", "time", "pressure_level"]:
            if (key in request_info) != (key in registered):
                return False
            if key in request_info and not set(request_info[key]) <= set(registered[key]):
                return False
        return True

    def download(self, client=None, **kwargs):

        cli = self.client 
        if client:
            cli = client # client overides self.client
        
        if not cli : 
            raise ValueError("Client is not set. configure CDS client before download request")

        # verify path
        cache_dir = Path(self.path)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # manage cache
        if self.clear_cache:
            print(cache_dir)
            # empty folder
            for element in cache_dir.iterdir():
                if element.is_dir():
                    shutil.rmtree(element)
                else:
                    element.unlink()

        # create a json with request fields
        # request on cds is defined by a product, + area + time
        if not self.request or self.original_request is None:
            raise ValueError("No request set yet, fill resquest form before download")

        request_info = {
            "dataset": self.product_info["cds_dataset"],
            "variables": self.original_request["variable"],
            "area": [float(a) for a in self.original_request["area"]],
            "state": "pending",     # pending / available / error
        }

        # time and pressure fields depend on the CDS dataset, not all of them have it
        for key in ["year", "month", "day", "time", "pressure_level"]:
            if key in self.original_request:
                request_info[key] = self.original_request[key]

        # register of uuid + associated parameters in cache dir, created if it does not exist
        register_file = cache_dir / "register.json"
        register = json.loads(register_file.read_text()) if register_file.exists() else {}

        # check if request correspond exactly or is include
        # in one of the ones set up in register. if it is, skip downloading
        for uuid_hex, registered in register.items():
            if registered["state"] == "available" and self._is_included(request_info, registered):
                # files already downloaded: the resolver reads them in their folder
                self._resolver = ExplicitFiles(cache_dir / uuid_hex)
                return Path(f"{cache_dir}/{uuid_hex}")

        # setup dedicated folder in cache dir based on
        cache = Path(f"{cache_dir}/{uuid.uuid4().hex}")
        cache.mkdir(parents=True, exist_ok=True)
        request_file = cache / "request.json"

        def register_update() -> None:
            """Write the request in its folder and in the register"""
            request_file.write_text(json.dumps(request_info, indent=4))
            register[cache.name] = request_info
            register_file.write_text(json.dumps(register, indent=4))

        register_update()

        # resolve request before download, divide it if necesary
        try:
            c=0
            for req in self.request:

                # remove if necessary
                # if os.path.exists(self.path):
                #     os.remove(self.path)
                retreive = cli.retrieve(self.product_info["cds_dataset"], req)

                # forge name of downloaded file
                forged_name = Path(retreive.location).name
                destination = Path(f"{cache}/{forged_name}")

                # actually download query
                retreive.download(destination)

                # if it is zip, unzip
                if Path(destination).suffix == ".zip":        
                    with zipfile.ZipFile(destination, "r") as zip_ref:
                        extracted_filenames = zip_ref.namelist()
                        zip_ref.extractall(destination.parent)

                    # rename files extracted from zip file
                    for file in extracted_filenames:
                        new_file = Path(f"{self.path}/{Path(retreive.location).stem}_{file}")
                        print("**  ", new_file)
                        Path(f"{self.path}/{file}").rename(new_file)

                    # remove zip file  
                    destination.unlink(missing_ok=True)


                c+=1

        except Exception:
            # refused by CDS or failed download: the original request is not available
            request_info["state"] = "error"
            register_update()
            raise

        # every sub-request downloaded: the original request is available
        request_info["state"] = "available"
        register_update()

        # the resolver reads the files of this request only, not the whole cache dir
        self._resolver = ExplicitFiles(cache)
        return cache

            
            



available_sources: dict[str, type[Source]] = {
    "local": LocalSource, 
    "remote-cds": CDSSource
    }
"""dict: Source classes by name, used by :class:`.Product`"""
