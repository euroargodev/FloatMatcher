# local_source.py:
#
# Two responsibilities, kept separate:
#   - FileResolver: "WHICH files do I open?" (from the points)
#   - LocalSource: opens the resolved files and returns the raw dataset.
#
# The orchestrator asks LocalSource for a dataset; normalization happens after,
# in the Product. LocalSource does NOT normalize and does NOT touch the network.

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.utils import SerializableLock

from .pointset import PointSet


# ─────────────────────────────────────────────────────────────
#  Path resolution (pure helper, no disk access)
# ─────────────────────────────────────────────────────────────

def resolve_path(root, pattern, date):
    """Substitute a date's components into `pattern` (under `root`). available fields are
    year, month, day, hour, minute (pick what the tree needs, with padding ex:
            "{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc").
    """
    ts = pd.Timestamp(date)
    rel = pattern.format(
        year=ts.year, month=ts.month, day=ts.day, hour=ts.hour, minute=ts.minute
    )
    return str(Path(root) / rel)


# ─────────────────────────────────────────────────────────────
#  Resolvers: which files for these points?
# ─────────────────────────────────────────────────────────────

class FileResolver(ABC):
    """Answers: which files must be opened for these points?"""

    @abstractmethod
    def files_for(self, points: PointSet) -> list[str]:
        ...


class ExplicitFiles(FileResolver):
    """Simplest resolver: the user lists the files explicitly."""

    def __init__(self, paths):
        if isinstance(paths, str):
            self._paths = [paths]  
        else:
            self._paths = list(paths)

    def files_for(self, points: PointSet) -> list[str]:
        return self._paths


class PathTemplate(FileResolver):
    """Resolve files from a declared directory pattern (e.g. Datarmor).

    The user declares the shape of their tree once; for each date PRESENT in the
    points, one path is built by substituting the date into the pattern. Only
    the dates actually needed are resolved (no directory scan). The field type
    (single-levels_inst, pressure-levels, ...) is a literal part of the pattern
    chosen by the user.

    Missing files are filtered out (with a warning): the user's time tolerance
    in the matchup, not the resolver, decides what gap is acceptable. But if
    NOTHING resolves to an existing file, that is a configuration error -> raise.
    """

    def __init__(self, root, pattern):
        self.root = root
        self.pattern = pattern

    def files_for(self, points: PointSet) -> list[str]:
        # unique days present in the points (day granularity; a hourly tree would
        # need datetime64[h] here — noted as a limitation for now)
        dates = np.unique(np.asarray(points.time).astype("datetime64[D]"))
        
        resolved = []
        for d in dates:
            resolved.append(resolve_path(self.root, self.pattern, d))

        existing = []
        missing = []
        for p in resolved:
            if Path(p).exists():
                existing.append(p)
            else:
                missing.append(p)

        if missing:
            warnings.warn(
                f"PathTemplate: {len(missing)} expected file(s) not found and "
                f"ignored (e.g. {missing[0]}). The matchup's time tolerance decides "
                f"whether the remaining dates are close enough."
            )
        if not existing:
            example = resolved[0] if resolved else "<no date in points>"
            raise FileNotFoundError(
                "PathTemplate: no file found for any of the requested dates. "
                f"Check `root` and `pattern` (a path resolved e.g. to: {example})."
            )
        return existing


# ─────────────────────────────────────────────────────────────
#  LocalSource: open the resolved files
# ─────────────────────────────────────────────────────────────

# netCDF4/HDF5 is not thread-safe, and open_mfdataset returns dask-backed
# arrays read by dask's threaded scheduler. ONE lock shared by every open
# serializes those reads process-wide; a per-call lock would not.
_NETCDF_LOCK = SerializableLock()


class LocalSource:
    """Opens files already present on disk and returns the raw dataset."""

    def __init__(self, resolver: FileResolver):
        self.resolver = resolver

    def open(self, points: PointSet) -> xr.Dataset:
        return self.open_paths(self.resolver.files_for(points))

    def open_paths(self, paths) -> xr.Dataset:
        """Open an explicit, already-resolved list of files and return the raw
        dataset. Bypasses the resolver: the caller has batched files_for(...)
        output into packets and opens each packet here.

        combine="by_coords" lets xarray order ERA5 files by their time coords;
        a single-element list works the same way.
        """
        if not paths:
            raise ValueError("LocalSource: empty file list")
        return xr.open_mfdataset(paths, combine="by_coords", lock=_NETCDF_LOCK)
