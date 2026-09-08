# nearest.py: nearest-neighbor matchup via separated spatial/temporal KDTrees.
#
# The spatial half is prepared ONCE (prepare) and reused on every temporal
# packet (match_packet), because the grid geometry is identical across packets.
# The packet loop itself lives in the orchestrator; this module provides the
# two halves and a single-pass `match` for the non-batched case.

import numpy as np

class NearestNeighbor():
    """Nearest-neighbor matchup method"""

    def __init__(self, max_dist_km: int = 25, 
                 max_time: np.timedelta64 = np.timedelta64(1, "D"),
                 k_nearest : int = 1) -> None :
        self.max_dist_km = max_dist_km
        self.max_time = max_time
        self.k_nearest = k_nearest

    @property
    def max_time_seconds(self) -> float:
        return float(self.max_time / np.timedelta64(1, "s"))
