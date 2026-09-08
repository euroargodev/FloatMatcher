# tests/helpers.py: helpers shared across the test folder
import numpy as np


def set_timestamps(*days):
    """Timestamps for the given 2015-01-xx days."""
    stamps = []
    for d in days:
        stamps.append(f"2015-01-{d:02d}")
    return np.array(stamps, dtype="datetime64[ns]")


def daily_timestamps(n):
    """n consecutive daily timestamps starting 2015-01-01."""
    return set_timestamps(*range(1, n + 1))
