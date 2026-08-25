# tests/test_reference.py
#
# ReferenceSet is now LAZY: grid_to_reference no longer materializes the value
# cube. It exposes read_values(node_idx, time_idx) to pull only the retained
# nodes. So the alignment tests read values through read_values instead of the
# old ref.values dict.

import numpy as np
import numpy.testing as npt
import xarray as xr

from floatmatcher.reference import ReferenceSet, grid_to_reference


def test_flatten_preserves_alignment():
    """Each flattened node's value matches its own coordinates.

    The grid value encodes position (1000*lat + lon). Reading every node via
    read_values must return, per node, exactly 1000*lat + lon — proving coords
    and values share the same flattening order.
    """
    lat = np.array([30.0, 31.0, 32.0])
    lon = np.array([-50.0, -49.0, -48.0, -47.0])
    lon2d, lat2d = np.meshgrid(lon, lat)
    field = 1000.0 * lat2d + lon2d
    ds = xr.Dataset({"v": (("lat", "lon"), field)}, coords={"lat": lat, "lon": lon})

    ref = grid_to_reference(ds)

    assert ref.lon.size == lat.size * lon.size            # 3 * 4 = 12

    all_nodes = np.arange(ref.lon.size)
    values = ref.read_values(all_nodes)["v"]              # 2D: no time_idx
    expected = 1000.0 * ref.lat + ref.lon
    npt.assert_allclose(values, expected, atol=1e-9)


def test_reference_is_2d_has_no_time():
    """A 2D grid produces a ReferenceSet with time=None."""
    lat = np.array([0.0, 1.0])
    lon = np.array([10.0, 11.0])
    ds = xr.Dataset({"v": (("lat", "lon"), np.zeros((2, 2)))},
                    coords={"lat": lat, "lon": lon})
    ref = grid_to_reference(ds)
    assert ref.time is None


def test_reference_xyz_shape():
    """xyz exposes one (x, y, z) row per node."""
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.array([10.0, 11.0])
    ds = xr.Dataset({"v": (("lat", "lon"), np.zeros((3, 2)))},
                    coords={"lat": lat, "lon": lon})
    ref = grid_to_reference(ds)
    assert ref.xyz.shape == (6, 3)                        # 3*2 nodes, 3 coords each


def test_flatten_3d_reads_correct_node_and_time():
    """A 3D grid: read_values pulls the right value for a given (node, time).

    Value encodes position AND time index (1000*lat + lon + 0.5*t), so reading
    all nodes at time 0 then time 1 must match the encoded values.
    """
    lat = np.array([30.0, 31.0])
    lon = np.array([-50.0, -49.0, -48.0])
    time = np.array([np.datetime64("2015-01-01"), np.datetime64("2015-01-02")])
    field = np.empty((time.size, lat.size, lon.size))
    for t in range(time.size):
        lon2d, lat2d = np.meshgrid(lon, lat)
        field[t] = 1000.0 * lat2d + lon2d + 0.5 * t
    ds = xr.Dataset({"v": (("time", "lat", "lon"), field)},
                    coords={"time": time, "lat": lat, "lon": lon})

    ref = grid_to_reference(ds)
    assert ref.time is not None

    nodes = np.arange(ref.lon.size)
    base = 1000.0 * ref.lat + ref.lon

    v_t0 = ref.read_values(nodes, np.zeros(nodes.size, dtype=int))["v"]
    npt.assert_allclose(v_t0, base, atol=1e-9)

    v_t1 = ref.read_values(nodes, np.ones(nodes.size, dtype=int))["v"]
    npt.assert_allclose(v_t1, base + 0.5, atol=1e-9)