# weights.py: weighting kernels for the interpolation methods.
#
# A kernel turns distances into UNNORMALIZED weights, and nothing else: it knows neither the
# radius nor the unit of what it weighs. The same function therefore serves the spatial axis
# (km) and the temporal one (seconds). Masking (radius, time gap) and normalization are the
# method's job, because normalization depends on the NaN mask of the data.

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def idw(d: ArrayLike, power: float = 1.0, limit: float | None = None,
        eps: float = 1e-3) -> NDArray[np.float64]:
    """Inverse distance weighting: ``1 / d**power``

    A point sitting exactly on a node gives an infinite weight. Distances below ``eps`` are
    clamped to it, so that node dominates the sum by orders of magnitude without overflowing.

    Parameters
    ----------
    d: array-like of float
        Distances, any shape. Not modified.
    power: float, default=1.0
        The higher it is, the more the closest neighbours dominate. ``power`` to infinity
        falls back on the nearest neighbour.
    limit: float, optional
        Unused, kept so that every kernel shares one signature.
    eps: float, default=1e-3
        Distance below which a node is taken to sit on the point. Far below the resolution
        of both axes (1 meter in km, 1 milisecond in seconds).

    Returns
    -------
    numpy.ndarray
        Unnormalized weights, same shape as ``d``.
    """
    dist = np.asarray(d, dtype=float)
    dist = np.where( dist < eps, eps, dist) # replace where dist < 1m or 1sec by 1 (avoid division by 0)
    weights = 1.0 / dist**power
    return weights



available_kernels: dict[str, Callable[..., NDArray[np.float64]]] = {
    "idw": idw,
}
"""dict: Weighting functions by name, used by :class:`.Interpolator`"""
