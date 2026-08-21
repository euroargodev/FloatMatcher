# tests/test_reference.py
import numpy as np
import numpy.testing as npt
import xarray as xr

from floatmatcher.reference import ReferenceSet, grid_to_reference


def test_flatten_preserves_alignment():
    """Each flattened node's value matches its own coordinates.

    The grid value encodes position (1000*lat + lon), so if coordinates and
    values were flattened in different orders, the check would fail.
    """
    lat = np.array([30.0, 31.0, 32.0])
    lon = np.array([-50.0, -49.0, -48.0, -47.0])
    lon2d, lat2d = np.meshgrid(lon, lat)          
    field = 1000.0 * lat2d + lon2d                
    ds = xr.Dataset(
        {"v": (("lat", "lon"), field)},
        coords={"lat": lat, "lon": lon},
    )

    ref = grid_to_reference(ds)

    # one node per grid cell
    assert ref.lon.size == lat.size * lon.size    # 3 * 4 = 12

    # the invariant: value[i] must equal 1000*lat[i] + lon[i]
    expected = 1000.0 * ref.lat + ref.lon
    npt.assert_allclose(ref.values["v"], expected, atol=1e-9)


def test_reference_is_2d_has_no_time():
    """A 2D grid produces a ReferenceSet with time=None."""
    lat = np.array([0.0, 1.0])
    lon = np.array([10.0, 11.0])
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.zeros((2, 2)))},
        coords={"lat": lat, "lon": lon},
    )
    ref = grid_to_reference(ds)
    assert ref.time is None


def test_reference_xyz_shape():
    """xyz exposes one (x, y, z) row per node."""
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([10.0, 11.0])
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.zeros((3, 2)))},
        coords={"lat": lat, "lon": lon},
    )
    ref = grid_to_reference(ds)
    assert ref.xyz.shape == (6, 3)                # 3*2 nodes, 3 coords each

def test_flatten_3d_keeps_time_dimension():
    """A 3D grid flattens space only; values keep a (node, time) shape."""
    lat = np.array([30.0, 31.0])
    lon = np.array([-50.0, -49.0, -48.0])
    time = np.array([np.datetime64("2015-01-01"), np.datetime64("2015-01-02")])
    # value encodes position AND time index: 1000*lat + lon + 0.5*time_index
    field = np.empty((time.size, lat.size, lon.size))
    for t in range(time.size):
        lon2d, lat2d = np.meshgrid(lon, lat)
        field[t] = 1000.0 * lat2d + lon2d + 0.5 * t
    ds = xr.Dataset(
        {"v": (("time", "lat", "lon"), field)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    ref = grid_to_reference(ds)

    assert ref.time is not None
    assert ref.values["v"].shape == (lat.size * lon.size, time.size)   # (node, time)
    # check a node/time entry matches its encoded position+time
    npt.assert_allclose(ref.values["v"][:, 0], 1000.0 * ref.lat + ref.lon, atol=1e-9)
    npt.assert_allclose(ref.values["v"][:, 1], 1000.0 * ref.lat + ref.lon + 0.5, atol=1e-9)