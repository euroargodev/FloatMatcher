# resolver.py:
#
# The resolver return a list of files to open, it doesn't open anything. 
# The resolver is split into a dummy class ExplicitFiles returning all files listed 
# or all files under a given path and a smarter one returning all files sorted along 
# a given pattern given by the user. 

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .pointset import PointSet

# ─────────────────────────────────────────────────────────────
#  Path resolution (pure helper, no disk access)
# ─────────────────────────────────────────────────────────────

def resolve_path(root: str | Path, pattern: str, date: Any) -> str:
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
#  Resolvers: which files to return ?
# ─────────────────────────────────────────────────────────────

class FileResolver(ABC):
    """Answers: which files must be opened for these points?"""

    @abstractmethod
    def files_for(self, points: PointSet | None = None) -> list[str]:
        ...


class ExplicitFiles(FileResolver):
    """Simplest resolver: the user lists the files explicitly 
    or a parent directory containing all files to process.
    """

    def __init__(self, paths: str | Path | Iterable[str | Path],
                 pattern: str = "*.nc") -> None:
        if isinstance(paths, (str, Path)):
            self._paths: list[str | Path] = [paths]
        else:
            self._paths = list(paths)
        self.pattern = pattern

    def files_for(self, points: PointSet | None = None) -> list[str]:
        resolved: list[str] = []
        for entry in self._paths:
            path = Path(entry)
            if path.is_dir():
                found = []
                for child in path.rglob(self.pattern):
                    found.append(str(child))
                found.sort()
                resolved.extend(found)
            else:
                resolved.append(str(path))
        return resolved


class PathTemplate(FileResolver):
    """Resolve files from a declared directory pattern (ex: Datarmor).

    The user declares the shape of their tree once; for each date PRESENT in the
    points, one path is built by substituting the date into the pattern. Only
    the dates actually needed are resolved (no directory scan). The field type
    (single-levels_inst, pressure-levels, ...) is a literal part of the pattern
    chosen by the user.

    Missing files are filtered out (with a warning): the user's time tolerance
    in the matchup, not the resolver, decides what gap is acceptable. But if
    NOTHING resolves to an existing file, that is a configuration error -> raise.
    """

    def __init__(self, root: str | Path, pattern: str,
                 granularity: Literal["Y", "M", "D", "h", "m", "s"] = "D", 
                 pad: int = 1) -> None:
        self.root = root
        self.pattern = pattern
        self.granularity = granularity
        self.pad = pad

    def files_for(self, points: PointSet | None = None) -> list[str]:
        if points is None:
            raise ValueError(
                "PathTemplate resolves one file per date present in the points; "
                "pass a PointSet, or use ExplicitFiles to list a directory."
            )

        dates = np.unique(
            np.asarray(points.time).astype(f"datetime64[{self.granularity}]")
        )
        
        # A point at 23h or 1h AM may well find its nearest step in the NEXT or PREVISOU file, so
        # the neighbours of every date are requested too.
        step = np.timedelta64(1, self.granularity)
        wanted = dates
        for shift in range(1, self.pad + 1):
            wanted = np.concatenate([wanted, dates - shift * step, dates + shift * step])
        wanted = np.unique(wanted)

        resolved = []
        for d in wanted:
            resolved.append(resolve_path(self.root, self.pattern, d))

        # Only the dates of the requested points are worth warning about:
        # padded neighbours are absent at the edges of any archive and does
        # not raise warnings.
        asked_for = set()
        for d in dates:
            asked_for.add(resolve_path(self.root, self.pattern, d))

        existing = []
        missing = []
        for p in resolved:
            if Path(p).exists():
                existing.append(p)
            elif p in asked_for:
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

