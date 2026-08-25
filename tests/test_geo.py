# tests/test_geo.py

import numpy as np
import numpy.testing as npt
import pytest



from floatmatcher.constants import EARTH_RADIUS_KM
from floatmatcher.geo import (
    lonlat_to_xyz,
    convert_lon,
    detect_lon_range,
    is_monotonic,
    is_strictly_increasing,
    is_global_lon
)


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
    xyz = lonlat_to_xyz([0, 10, 20], [0, 10, 20])
    assert xyz.shape == (3, 3)


def test_points_lie_on_sphere():
    """every point is at the same distance from the center"""
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, size=100)
    lat = rng.uniform(-90, 90, size=100)
    xyz = lonlat_to_xyz(lon, lat)

    norms = np.linalg.norm(xyz, axis=1)
    npt.assert_allclose(norms, EARTH_RADIUS_KM, atol=1e-9)


def test_antimeridian_points_are_close():
    """2 points near -180 and 180 are close in 3D"""
    xyz = lonlat_to_xyz([179, -179], [0, 0])
    dist = np.linalg.norm(xyz[0] - xyz[1])
    # equator: 2° ~222 km 
    assert dist < 250



# ───────────── convert_lon ─────────────
 
def test_convert_lon_to_180():
    npt.assert_allclose(convert_lon([0, 90, 270, 350], "-180-180"),
                        [0, 90, -90, -10])
 
 
def test_convert_lon_to_360():
    npt.assert_allclose(convert_lon([-45, -10, 0, 179], "0-360"),
                        [315, 350, 0, 179])
 
 
def test_convert_lon_idempotent():
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
 
 
# ───────────── detect_lon_range ─────────────
 
def test_detect_global_0_360():
    assert detect_lon_range(np.arange(0, 360)) == "0-360"
 
 
def test_detect_standard_180():
    assert detect_lon_range(np.arange(-180, 180)) == "-180-180"
 
 
def test_detect_regional_west_360():
    assert detect_lon_range([300, 310, 350]) == "0-360"        # max > 180
 
 
def test_detect_regional_east_180():
    assert detect_lon_range([-40, -30, -10]) == "-180-180"     # min < 0
 
 
def test_detect_ambiguous_defaults_180():
    # all in [0,180] -> conventions coincide -> default -180-180
    assert detect_lon_range([10, 50, 170]) == "-180-180"
 
 
# ───────────── is_monotonic (both directions accepted) ─────────────
 
def test_monotonic_increasing():
    assert is_monotonic([-180, 0, 90]) is True
 
 
def test_monotonic_decreasing_accepted():
    # ERA5 latitude often runs north -> south
    assert is_monotonic([90, 45, 0, -45, -90]) is True
 
 
def test_monotonic_disorder_rejected():
    assert is_monotonic([0, 90, -90, 180]) is False
 
 
def test_monotonic_duplicate_rejected():
    assert is_monotonic([0, 10, 10, 20]) is False
 
 
def test_monotonic_single_point():
    assert is_monotonic([5.0]) is True
 
 
def test_monotonic_on_datetime():
    t = np.array(["2015-01-01", "2015-01-02"], dtype="datetime64[ns]")
    assert is_monotonic(t) is True
 
 
# ───────────── is_strictly_increasing (time axis: increasing only) ─────────────
 
def test_increasing_true():
    t = np.array(["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]")
    assert is_strictly_increasing(t) is True
 
 
def test_increasing_rejects_decreasing():
    # THE difference with is_monotonic: decreasing is NOT accepted for time
    t = np.array(["2015-01-03", "2015-01-02", "2015-01-01"], dtype="datetime64[ns]")
    assert is_strictly_increasing(t) is False
 
 
def test_increasing_rejects_duplicate():
    t = np.array(["2015-01-01", "2015-01-01"], dtype="datetime64[ns]")
    assert is_strictly_increasing(t) is False
 
 
def test_increasing_single_point():
    assert is_strictly_increasing([5.0]) is True
 
 
# ───────────── lonlat_to_xyz : convention invariance ─────────────
# This is what lets nearest-neighbour ignore the longitude convention.
 
def test_xyz_convention_invariant():
    a = lonlat_to_xyz([350.0], [30.0])   # 350 == -10 : same meridian
    b = lonlat_to_xyz([-10.0], [30.0])
    npt.assert_allclose(a, b, atol=1e-9)
 

# ───────────── PAD circular longitudes ─────────────

def test_is_global_0_360():
    assert is_global_lon(np.arange(0, 360, 0.25)) is True
 
 
def test_is_global_minus180_180():
    assert is_global_lon(np.arange(-180, 180, 0.25)) is True
 
 
def test_is_global_regional_is_false():
    # an Atlantic cut-out does not wrap the globe
    assert is_global_lon(np.arange(-100, 20, 0.25)) is False
 
 
def test_is_global_single_point_is_false():
    assert is_global_lon(np.array([10.0])) is False
 
 
def test_is_global_coarse_but_global():
    # coarse yet global: step 90, span 360 once the wrap cell is added
    assert is_global_lon(np.array([0.0, 90.0, 180.0, 270.0])) is True
