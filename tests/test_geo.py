# tests/test_geo.py

import numpy as np
import numpy.testing as npt
import pytest
from floatmatcher.constants import EARTH_RADIUS_KM
from floatmatcher.geo import (
    lonlat_to_xyz
)


# ───────────── lonlat_to_xyz ─────────────

def test_reference_points():
    """ set special points to verify the computations """
    lon = [0, 90, 0, 0]
    lat = [0, 0, 90, -90]
    xyz = lonlat_to_xyz(lon, lat)

    expected = np.array([
        [EARTH_RADIUS_KM, 0, 0],     # equator / Greenwich
        [0, EARTH_RADIUS_KM, 0],     # equator / 90°E
        [0, 0, EARTH_RADIUS_KM],     # north pole
        [0, 0, -EARTH_RADIUS_KM],    # south pole
    ])
    npt.assert_allclose(xyz, expected, atol=1e-9)


def test_output_shape():
    """N points entry -> output (N, 3)"""
    xyz = lonlat_to_xyz([0, 10, 20, 30], [0, 10, 20, 30])
    assert xyz.shape == (4, 3)


def test_points_lie_on_sphere():
    """every point is at the same distance from the center"""
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, size=100)
    lat = rng.uniform(-90, 90, size=100)
    xyz = lonlat_to_xyz(lon, lat)

    norms = np.linalg.norm(xyz, axis=1) # compute norm of each xyz lon/lat vector 
    npt.assert_allclose(norms, EARTH_RADIUS_KM, atol=1e-9)


def test_antimeridian_points_are_close():
    """2 points near -180 and 180 are close in 3D"""
    xyz = lonlat_to_xyz([179, -179], [0, 0])
    dist = np.linalg.norm(xyz[0] - xyz[1])
    # equator: 2° ~222 km 
    assert dist < 250


def test_xyz_convention_invariant():
    """each convention 0-360 and -180-180 must give the same xyz"""
    a = lonlat_to_xyz([350.0], [30.0])   # 350 == -10 : same meridian
    b = lonlat_to_xyz([-10.0], [30.0])
    npt.assert_allclose(a, b, atol=1e-9)

