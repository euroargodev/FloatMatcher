# method.py: the matchup method contract (abstract base class)

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .gridset import GridSet
from .pointset import PointSet
from .results import MatchupResult


@dataclass
class Constraints:
    """Colocalization limits: a matched neighbor beyond these is rejected."""

    max_dist_km: float = 100.0
    max_time_days: float = 1.0


class MatchupMethod(ABC):
    """Contract for all matchup methods.

    Any method (nearest neighbor, interpolation, ...) must implement ``match``
    with this exact signature, so that methods are interchangeable and a new
    one can be added without touching the rest of the engine.
    """

    @abstractmethod
    def match(
        self,
        grid: GridSet,
        points: PointSet,
        constraints: Constraints,
    ) -> MatchupResult:
        """Colocalize ``points`` against ``grid`` and return a MatchupResult."""
        ...