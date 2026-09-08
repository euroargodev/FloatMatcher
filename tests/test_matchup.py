# tests/test_matchup.py

import numpy as np

from floatmatcher.matchup import NearestNeighbor


def test_default_tolerance_is_one_day_in_seconds():
    assert NearestNeighbor().max_time_seconds == 86400.0


def test_max_time_accepts_any_unit():
    assert NearestNeighbor(max_time=np.timedelta64(3, "h")).max_time_seconds == 10800.0
    assert NearestNeighbor(max_time=np.timedelta64(90, "m")).max_time_seconds == 5400.0
    assert NearestNeighbor(max_time=np.timedelta64(2, "D")).max_time_seconds == 172800.0
