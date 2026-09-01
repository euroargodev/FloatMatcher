# constants.py: project-wide constants and conventions, gathered in one place.

import numpy as np

# --- geography ---
EARTH_RADIUS_KM: float = 6371.0
"""Mean Earth radius (km), used to project lon/lat onto a sphere in geo.py."""

# --- time ---
TIME_UNIT: str = "ns"
"""Canonical datetime64 resolution enforced across the library (points and grids)."""

REF_TIME: np.datetime64 = np.datetime64("1970-01-01", "ns")
"""Reference epoch: datetime64 values are turned into floating-point days from
here for the 1D temporal KDTree. Any fixed epoch works; what matters is that
points and grid use the SAME one, so their difference is correct."""
