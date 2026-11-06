# sources.py: where the product files come from : local/remote

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import copy
import json
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
            self.original_request: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return _indented_repr(self)

    def build_request(self, points: PointSet,
                      variables: list[str] | None) -> list[dict[str, Any]]:
        """Request of the points for a remote source, only ``"remote-cds"`` has one

        Parameters
        ----------
        points: :class:`.PointSet`
        variables: list of str or None

        Raises
        ------
        ValueError
            Always, the source has no request. See :meth:`.CDSSource.build_request`.
        """
        raise ValueError("request can be build only if source == remote-cds")


    def download(self, **kwargs: Any) -> Path:
        """Download the last request built, only ``"remote-cds"`` has one

        Raises
        ------
        ValueError
            Always, the source has nothing to download. See :meth:`.CDSSource.download`.
        """
        raise ValueError("download is available only if source == remote-cds")

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
    def fetch(self, points: PointSet, variables: list[str] | None) -> None:
        """Make the needed files available locally, downloading them if remote

        Parameters
        ----------
        points: :class:`.PointSet`
        variables: list of str or None
            Names in the files of the variables needed. None keeps all, a remote source
            needs a list.
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

    def fetch(self, points: PointSet, variables: list[str] | None)-> None:
        """Nothing to do, the files are already on disk"""
        pass
        

class CDSSource(Source):
    """Files downloaded from the Copernicus Climate Data Store (CDS)

    The request is built from the points (area, dates, variables), split under the cost limit
    of CDS, then downloaded in a folder of the cache directory. A register of the requests
    already downloaded, ``register.json`` in the cache directory, avoids downloading again a
    request included in a previous one.

    Parameters
    ----------
    path: str or pathlib.Path
        Cache directory. Each request gets its own folder in it, named by a uuid.
    product_info: dict
        Information of the product: ``cds_dataset``, ``cds_request``, ``cds_variables``,
        ``cds_options``, ``grid_step``, ``lon_convention``, ``time_resolution`` and
        ``time_margin``.
    client: cdsapi.Client or mocking.FakeCDSClient
        Client of the CDS API. :class:`.FakeCDSClient` to develop without network.
    clear_cache: bool, default False
        Empty the cache directory before downloading.
    **kwargs: list of str
        Values of the options of the product (ERA5: ``time``, e.g. ``["00:00", "12:00"]``),
        all values if not given. Kept for every request built.

    Raises
    ------
    ValueError
        If an option is unknown, or its values are empty or not available.

    Attributes
    ----------
    options: dict
        Option -> values asked, given at creation.
    original_request: dict or None
        Last request built from the points, before the split.
    request: list of dict or None
        Sub-requests of ``original_request``, each one under the cost limit.
    """
    def __init__(self,
                 path: str | Path,
                 product_info: dict[str, Any],
                 client: Any,
                 clear_cache: bool = False,
                 **kwargs: list[str]
                 ) -> None:
        
        self.path: str | Path = path      # cache dir, a single directory
        self.product_info: dict[str, Any] = product_info
        self.client = client      # cdsapi.Client, or mocking.FakeCDSClient to develop without network
        self.clear_cache = clear_cache

        # options of the requests, checked now so that a wrong one fails at set_product
        self._options_request(kwargs)
        self.options: dict[str, list[str]] = kwargs


        # in remote case, resolver is the cache dir given in path, until download()
        # points it to the folder of the request
        self._resolver: FileResolver = ExplicitFiles(self.path)
        
        # request template of the product, completed from the points by build_request
        self.base_request: dict[str, Any] = copy.deepcopy(product_info["cds_request"])
        # last request built, and its sub-requests under the cost limit of CDS
        self.original_request: dict[str, Any] | None = None
        self.request: list[dict[str, Any]] | None = None


    def resolve(self, points: PointSet) -> list[str]:
        """ Files needed for these points """
        return self._resolver.files_for(points)

    def fetch(self, points: PointSet, variables: list[str] | None)-> None:
        """Download the files of the points from CDS, if they are not in the cache yet

        Creates the cache directory, then calls :meth:`build_request` and :meth:`download`.
        Called by ``Product.open``, so by ``match()``.

        Parameters
        ----------
        points: :class:`.PointSet`
        variables: list of str
            Names in the files, keys of ``product_info["cds_variables"]``.

        Raises
        ------
        ValueError
            If the request can not be built (see :meth:`build_request`).
        requests.HTTPError
            If CDS refuses a sub-request.
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

    def _area(self, points: PointSet) -> list[float]:
        """Area covering the points plus a margin, rounded outward on the grid of the product

        Longitudes follow ``product_info["lon_convention"]``, ``"-180/180"`` or ``"0/360"``. If
        the box crosses the edge of the convention (the antimeridian in -180/180, Greenwich in
        0/360), the whole band of latitude is asked: one area, no split.

        Parameters
        ----------
        points: :class:`.PointSet`

        Returns
        -------
        list of float
            ``[north, west, south, east]``, the order CDS expects.
        """
        # degrees, at least the matchup radius (0.5° is about 55 km)
        margin = self.product_info["grid_step"] *2
        grid_step = self.product_info["grid_step"]        # degrees, resolution of ERA5

        # longitude convention of the product
        lon_min = {"-180/180": -180.0, "0/360": 0.0}[self.product_info["lon_convention"]]
        lon_max = lon_min + 360

        south = max(np.floor((points.lat.min() - margin) / grid_step) * grid_step, -90)
        north = min(np.ceil((points.lat.max() + margin) / grid_step) * grid_step, 90)

        # smallest arc covering the points, whatever the convention of their longitudes: the
        # globe minus the largest gap between two successive points
        lon = np.sort((np.asarray(points.lon) - lon_min) % 360 + lon_min)
        gaps = np.diff(np.append(lon, lon[0] + 360))
        i = int(np.argmax(gaps))
        if i == lon.size - 1:
            # largest gap across the edge: the arc does not cross it
            west, east = lon[0], lon[-1]
        else:
            west, east = lon[i + 1], lon[i] + 360

        west = np.floor((west - margin) / grid_step) * grid_step
        east = np.ceil((east + margin) / grid_step) * grid_step

        # the box (margin included) crosses the edge, or goes round the globe: whole band of
        # latitude. The east edge one step before lon_max, the same meridian as lon_min
        if west < lon_min or east > lon_max:
            return [float(north), lon_min, float(south), lon_max - grid_step]
        return [float(north), float(west), float(south), float(east)]

    def _build_request(self, points: PointSet,
                      variables: list[str] | None) -> dict[str, Any]:
        """Request for the points, from the request template of the product

        Area of the points plus a margin (the whole band of latitude if it crosses the edge of
        the longitudes, see ``_area``), variables translated to the CDS names, options of the
        product, and the days of the points with ``time_margin`` around them.

        Parameters
        ----------
        points: :class:`.PointSet`
        variables: list of str
            Names in the files, keys of ``product_info["cds_variables"]``.

        Returns
        -------
        dict
            Request in the form CDS expects.

        Raises
        ------
        ValueError
            If ``variables`` is None or has unknown names, if an option is unknown or has
            values not available, or if ``time_margin`` is 24 h or more.
        """
        # copy initial request
        request = copy.deepcopy(self.base_request)

        # Variables: names in the files, translated to the names CDS expects
        cds_names = self.product_info["cds_variables"]
        if variables is None:
            raise ValueError(f"CDS needs a list of variables, available: {list(cds_names)}")
        unknown = [v for v in variables if v not in cds_names]
        if unknown:
            raise ValueError(f"unknown variables {unknown}, available: {list(cds_names)}")
        request["variable"] = [cds_names[v] for v in variables]

        # Options of the source (ERA5: time), all values if not given
        request.update(self._options_request(self.options))

        # time creation : 
        # par default, le time resolution = resolution de la grille min (pour ERA5= 1h)
        dates = np.unique(
                    np.asarray(points.time).astype(f"datetime64[{self.product_info['time_resolution']}]")
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

        # area of the points, the whole band of latitude if it crosses the edge of the longitudes
        request["area"] = self._area(points)      # CDS expects this order

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
        costs: dict[str, Any] = self.client.client.estimate_costs(dataset_name, request)

        return costs

    def build_request(self, points: PointSet,
                      variables: list[str] | None) -> list[dict[str, Any]]:
        """Request of the points, split under the cost limit of CDS

        The request is split by year, then by month for the years still too large. The
        request and its sub-requests are kept in ``original_request`` and ``request``.
        The options are the ones of the source, given at its creation.

        Parameters
        ----------
        points: :class:`.PointSet`
        variables: list of str
            Names in the files, keys of ``product_info["cds_variables"]``.

        Returns
        -------
        list of dict
            Sub-requests to download, the request alone if it fits in the limit.

        Raises
        ------
        ValueError
            If the request can not be built (see ``_build_request``), or if a single month
            exceeds the limit.
        """
        # build first request
        self.original_request = self._build_request(points, variables)

        # eval cost of original request
        req_costs = self.eval_request(self.original_request)

        # subdivide request if necessary,
        if req_costs["cost"] >= req_costs["limit"]:
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
                        # even a single month is not downloadable
                        raise ValueError("a single month exceeds limit, consider reduce area or variables "
                                         "or hours")
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

    def download(self, client: Any = None, **kwargs: Any) -> Path:
        """Download the sub-requests of the last request built, in a folder of the cache

        The request is written in the register (``register.json`` in the cache directory) and
        in ``request.json`` in its folder, with a state: ``pending`` while downloading, then
        ``available``, or ``error`` if CDS refuses it, the download fails or is interrupted.
        Not downloaded if the request is included in an ``available`` request of the register.
        A same request in ``error`` is removed, folder included, and downloaded again. Zip files
        are extracted. The resolver then reads the files of this folder only.

        Parameters
        ----------
        client: cdsapi.Client 

        Returns
        -------
        pathlib.Path
            Folder of the files of the request.

        Raises
        ------
        ValueError
            If there is no client, or no request built yet.
        requests.HTTPError
            If CDS refuses a sub-request.
        """
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
            # empty folder
            for element in cache_dir.iterdir():
                if element.is_dir():
                    shutil.rmtree(element)
                else:
                    element.unlink()

        if not self.request or self.original_request is None:
            raise ValueError("No request set yet, fill resquest form before download")

        # register of uuid + associated parameters in cache dir, created if it does not exist
        register_file = Path(f"{cache_dir}/register.json")
        register = json.loads(register_file.read_text()) if register_file.exists() else {}

        # create a json with request fields
        # request on cds is defined by a product, + area + time
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

        # same request already failed: remove its folder and entry, then download it fresh
        for uuid_hex, registered in list(register.items()):
            if (registered["state"] == "error" and self._is_included(request_info, registered)
                    and self._is_included(registered, request_info)):
                shutil.rmtree(Path(f"{cache_dir}/{uuid_hex}"), ignore_errors=True)
                del register[uuid_hex]
        register_file.write_text(json.dumps(register, indent=4))

        # check if request correspond exactly or is include
        # in one of the ones set up in register. if it is, skip downloading
        for uuid_hex, registered in register.items():
            if registered["state"] == "available" and self._is_included(request_info, registered):
                # files already downloaded: the resolver reads them in their folder
                self._resolver = ExplicitFiles(Path(f"{cache_dir}/{uuid_hex}"))
                return Path(f"{cache_dir}/{uuid_hex}")

        # setup dedicated folder in cache dir based on
        cache = Path(f"{cache_dir}/{uuid.uuid4().hex}")
        cache.mkdir(parents=True, exist_ok=True)
        request_file = Path(f"{cache}/request.json")

        def register_update() -> None:
            """Write the request in its folder and in the register"""
            request_file.write_text(json.dumps(request_info, indent=4))
            register[cache.name] = request_info
            register_file.write_text(json.dumps(register, indent=4))

        register_update()

        # download the sub-requests
        try:
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
                if destination.suffix == ".zip":
                    with zipfile.ZipFile(destination, "r") as zip_ref:
                        extracted_filenames = zip_ref.namelist()
                        zip_ref.extractall(cache)

                    # rename files extracted from zip file, prefixed by the zip name: the zips of
                    # the sub-requests hold files with the same names
                    for file in extracted_filenames:
                        Path(f"{cache}/{file}").rename(Path(f"{cache}/{destination.stem}_{file}"))

                    # remove zip file
                    destination.unlink(missing_ok=True)

        except BaseException:
            # refused by CDS, failed download or interrupted (Ctrl+C): the original request
            # is not available
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
