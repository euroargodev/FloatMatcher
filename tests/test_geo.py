# tests/test_geo.py

import numpy as np
import numpy.testing as npt

from floatmatcher.geo import lonlat_to_xyz

R = 6371.0

def test_reference_points():
    """ set special points to verify the computations """
    lon = [0, 90, 0, 0]
    lat = [0, 0, 90, -90]
    xyz = lonlat_to_xyz(lon, lat)

    expected = np.array([
        [R, 0, 0],     # equator / Greenwich
        [0, R, 0],     # equator / 90°E
        [0, 0, R],     # north pole
        [0, 0, -R],    # south pole
    ])
    npt.assert_allclose(xyz, expected, atol=1e-9)


def test_output_shape():
    """N points entry -> output (N, 3)"""
    xyz = lonlat_to_xyz([0, 10, 20], [0, 10, 20])
    assert xyz.shape == (3, 3)


def test_points_lie_on_sphere():
    """every point is at the same distance from the center"""
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, size=100)
    lat = rng.uniform(-90, 90, size=100)
    xyz = lonlat_to_xyz(lon, lat)

    norms = np.linalg.norm(xyz, axis=1)
    npt.assert_allclose(norms, R, atol=1e-9)


def test_antimeridian_points_are_close():
    """2 points near -180 and 180 are close in 3D"""
    xyz = lonlat_to_xyz([179, -179], [0, 0])
    dist = np.linalg.norm(xyz[0] - xyz[1])
    # equator: 2° ~222 km 
    assert dist < 250