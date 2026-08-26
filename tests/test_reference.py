# tests/test_reference.py
#
# ReferenceSet is now LAZY: grid_to_reference no longer materializes the value
# cube. It exposes read_values(node_idx, time_idx) to pull only the retained
# nodes. So the alignment tests read values through read_values instead of the
# old ref.values dict.

import numpy as np
import numpy.testing as npt

from floatmatcher.reference import grid_to_reference

from helpers import make_grid, pos_field, daily_timestamps


def test_flatten_preserves_alignment():
    """Each flattened node's value matches its own coordinates.

    The grid value encodes position (1000*lat + lon). Reading every node via
    read_values must return, per node, exactly 1000*lat + lon — proving coords
    and values share the same flattening order.
    """
    ds = make_grid([30.0, 31.0, 32.0], [-50.0, -49.0, -48.0, -47.0],
                   fill=pos_field, fill_args=(1000.0, 0.0))

    ref = grid_to_reference(ds)

    assert ref.lon.size == 3 * 4                          # 3 * 4 = 12

    all_nodes = np.arange(ref.lon.size)
    values = ref.read_values(all_nodes)["v"]              # 2D: no time_idx
    expected = 1000.0 * ref.lat + ref.lon
    npt.assert_allclose(values, expected, atol=1e-9)


def test_reference_is_2d_has_no_time():
    """A 2D grid produces a ReferenceSet with time=None."""
    ds = make_grid([0.0, 1.0], [10.0, 11.0])
    ref = grid_to_reference(ds)
    assert ref.time is None


def test_reference_xyz_shape():
    """xyz exposes one (x, y, z) row per node."""
    ds = make_grid([0.0, 1.0, 2.0], [10.0, 11.0])
    ref = grid_to_reference(ds)
    assert ref.xyz.shape == (6, 3)                        # 3*2 nodes, 3 coords each


def test_flatten_3d_reads_correct_node_and_time():
    """A 3D grid: read_values pulls the right value for a given (node, time).

    Value encodes position AND time index (1000*lat + lon + 0.5*t), so reading
    all nodes at time 0 then time 1 must match the encoded values.
    """
    ds = make_grid([30.0, 31.0], [-50.0, -49.0, -48.0],
                   time=daily_timestamps(2),
                   fill=pos_field, fill_args=(1000.0, 0.5))

    ref = grid_to_reference(ds)
    assert ref.time is not None

    nodes = np.arange(ref.lon.size)
    base = 1000.0 * ref.lat + ref.lon

    v_t0 = ref.read_values(nodes, np.zeros(nodes.size, dtype=int))["v"]
    npt.assert_allclose(v_t0, base, atol=1e-9)

    v_t1 = ref.read_values(nodes, np.ones(nodes.size, dtype=int))["v"]
    npt.assert_allclose(v_t1, base + 0.5, atol=1e-9)