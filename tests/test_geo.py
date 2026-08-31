# tests/test_geo.py

import numpy as np
import numpy.testing as npt
import pytest



from floatmatcher.constants import EARTH_RADIUS_KM
from floatmatcher.geo import (
    lonlat_to_xyz,
    convert_lon,
    detect_lon_range
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

# This is what lets nearest-neighbour ignore the longitude convention.
# each convention 0-360 and -180-180 must give the same xyz
def test_xyz_convention_invariant():
    a = lonlat_to_xyz([350.0], [30.0])   # 350 == -10 : same meridian
    b = lonlat_to_xyz([-10.0], [30.0])
    npt.assert_allclose(a, b, atol=1e-9)


# ───────────── convert_lon ─────────────
 
def test_convert_lon_to_180():
    npt.assert_allclose(convert_lon([0, 90, 270, 350], "-180-180"),
                        [0, 90, -90, -10])
 
 
def test_convert_lon_to_360():
    npt.assert_allclose(convert_lon([-45, -10, 0, 179], "0-360"),
                        [315, 350, 0, 179])
 
 
def test_convert_lon_doing_nothing():
    a180 = [-45.0, 0.0, 45.0]
    npt.assert_allclose(convert_lon(a180, "-180-180"), a180)
    a360 = [10.0, 200.0, 359.0]
    npt.assert_allclose(convert_lon(a360, "0-360"), a360)
 
 
def test_convert_lon_preserves_order():
    # conversion must NOT sort: same positions, only relabelled
    npt.assert_allclose(convert_lon([350, 10, 200], "-180-180"), [-10, 10, -160])
 
 
def test_convert_lon_unknown_range_raises():
    with pytest.raises(ValueError):
        convert_lon([0, 1, 2], "0-180")
 
 
# # ───────────── detect_lon_range ─────────────
 
# def test_detect_global_0_360():
#     assert detect_lon_range(np.arange(0, 360)) == "0-360"
 
 
# def test_detect_standard_180():
#     assert detect_lon_range(np.arange(-180, 180)) == "-180-180"
 
 
# def test_detect_regional_west_360():
#     assert detect_lon_range([300, 310, 350]) == "0-360"        # max > 180
 
 
# def test_detect_regional_east_180():
#     assert detect_lon_range([-40, -30, -10]) == "-180-180"     # min < 0
 
 
# def test_detect_ambiguous_defaults_180():
#     # all in [0,180] -> conventions coincide -> default -180-180
#     assert detect_lon_range([10, 50, 170]) == "-180-180"
 
 