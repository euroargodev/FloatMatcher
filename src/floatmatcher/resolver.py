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

def _resolve_path(root: str | Path, pattern: str, date: Any) -> str:
    """Substitute the components of a date into a path template, under a root

    Pure helper, no disk access. The fields are ``{year}``, ``{month}``, ``{day}``, ``{hour}``
    and ``{minute}``: pick what the tree needs, with padding if wanted.

    Parameters
    ----------
    root: str or pathlib.Path
    pattern: str
        Path template, e.g. ``"{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc"``.
    date: datetime-like
        Anything :class:`pandas.Timestamp` accepts.

    Returns
    -------
    str
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
    """Prototype of the resolvers: which files to open for these points"""

    @abstractmethod
    def files_for(self, points: PointSet | None = None) -> list[str]:
        """Files to open for these points

        Parameters
        ----------
        points: :class:`PointSet`, optional
            Positions to match. Some resolvers do not need them.

        Returns
        -------
        list of str
        """
        ...


class ExplicitFiles(FileResolver):
    """Files listed by the user, or found in a directory

    Parameters
    ----------
    paths: str, pathlib.Path or iterable of them
        Files, or directories scanned recursively.
    pattern: str, default="*.nc"
        Name pattern of the files looked for in a directory.
    """

    def __init__(self, paths: str | Path | Iterable[str | Path],
                 pattern: str = "*.nc") -> None:
        if isinstance(paths, (str, Path)):
            self._paths: list[str | Path] = [paths]
        else:
            self._paths = list(paths)
        self.pattern = pattern

    def files_for(self, points: PointSet | None = None) -> list[str]:
        """Files listed, or the sorted files found in the directories

        The points are not used.

        Returns
        -------
        list of str
        """
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
    """Files built from a path template (e.g. on Datarmor)

    One path is built for each date of the points, by substituting the date into ``pattern``.
    Only the dates needed are resolved, with no directory scan. Neighboring dates are requested
    too, as a point can find its nearest time step in the next or previous file.

    Missing files are skipped with a warning: the time gap of the matchup decides what is
    acceptable. If no file exists at all, it is a configuration error.

    Parameters
    ----------
    root: str or pathlib.Path
        Root of the tree.
    pattern: str
        Path template under ``root``, with the fields ``{year}``, ``{month}``, ``{day}``,
        ``{hour}`` and ``{minute}``.
    granularity: {'Y', 'M', 'D', 'h', 'm', 's'}, default='D'
        Time step of the files, one file per step.
    pad: int, default=1
        Number of neighboring steps requested on each side of every date.
    """

    def __init__(self, root: str | Path, pattern: str,
                 granularity: Literal["Y", "M", "D", "h", "m", "s"] = "D", 
                 pad: int = 1) -> None:
        self.root = root
        self.pattern = pattern
        self.granularity = granularity
        self.pad = pad

    def files_for(self, points: PointSet | None = None) -> list[str]:
        """Existing files for the dates of the points

        Parameters
        ----------
        points: :class:`PointSet`
            Required: its times give the dates.

        Returns
        -------
        list of str

        Raises
        ------
        ValueError
            If ``points`` is None or has no time.
        FileNotFoundError
            If none of the files exists.

        Warns
        -----
        UserWarning
            If the file of a date of the points is missing.
        """
        if points is None:
            raise ValueError(
                "PathTemplate resolves one file per date present in the points; "
                "pass a PointSet, or use ExplicitFiles to list a directory."
            )

        dates = np.unique(
            np.asarray(points.time).astype(f"datetime64[{self.granularity}]")
        )
        
        # A point at 23h or 1h AM may well find its nearest step in the NEXT or PREVIOUS file, so
        # the neighbors of every date are requested too.
        step = np.timedelta64(1, self.granularity)
        wanted = dates
        for shift in range(1, self.pad + 1):
            wanted = np.concatenate([wanted, dates - shift * step, dates + shift * step])
        wanted = np.unique(wanted)

        resolved = []
        for d in wanted:
            resolved.append(_resolve_path(self.root, self.pattern, d))

        # Only the dates of the requested points are worth warning about:
        # padded neighbors are absent at the edges of any archive and does
        # not raise warnings.
        asked_for = set()
        for d in dates:
            asked_for.add(_resolve_path(self.root, self.pattern, d))

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

