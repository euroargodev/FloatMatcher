# sources.py: where the product files come from : local/remote

from abc import ABC, abstractmethod
from pathlib import Path

from .pointset import PointSet
from .resolver import FileResolver, PathTemplate, ExplicitFiles
from .utils import indented_repr


class Source(ABC):

    def __init__(self, 
                 path: str | Path | list[str]
                 ) -> None:
            self.path = path


    def __repr__(self) -> str:
        return indented_repr(self)

    @abstractmethod
    def resolve(self, points: PointSet) -> list[str]:
        """Which files are needed for these points."""

    @abstractmethod
    def fetch(self, points: PointSet) -> None:
        """Make the needed files available locally (download if remote)."""


class LocalSource(Source):
    def __init__(self,
                 path: str | list[str] | Path,
                 pattern: str | None =None) -> None:
        
        self.path = path
        self.pattern = pattern

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
        return self._resolver.files_for(points)

    def fetch(self, points: PointSet)-> None:
        pass
        


available_sources: dict[str, type[Source]] = {
    "local": LocalSource
    }
