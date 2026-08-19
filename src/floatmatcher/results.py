# results.py: output object of matchup

import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class MatchupResult:
    values:      dict[str, NDArray[np.float64]]   
    distance_km: NDArray[np.float64]        
    time_delta:  NDArray[np.float64]         
    valid:       NDArray[np.bool_]             

    def to_dataset(ds, points:PointSet):
        dim = points.origin_dim
        if not dim:         # No original dataset --> nonsens to create a dataset --> 
                            # no, TODO : change behaviour let the user decide if it worth a xarraydataset or not
            return ""
        else:
            if ds.sizes[dim] != len(self.valid):
                raise ValueError("original dataset has not the same length than matchup")

        