# tests/test_accessors.py: argopy extension ds.argo.floatmatcher

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

@pytest.fixture
def argo_profiles():
    """Two profiles of three levels, in the argopy N_PROF layout

    Each profile sits exactly on a node of grid_3d_ds, where sst = 100 * lat + lon + t:

    profile   lon   lat   time (t)          expected sst_coloc
    0         20    1     2015-01-01 (0)    100 * 1 + 20 + 0 = 120
    1         30    2     2015-01-02 (1)    100 * 2 + 30 + 1 = 231
    """
    return xr.Dataset(
        {
            "PLATFORM_NUMBER": ("N_PROF", [6901234, 6901234]),
            "CYCLE_NUMBER": ("N_PROF", [1, 2]),
            "DIRECTION": ("N_PROF", ["A", "A"]),
            "LONGITUDE": ("N_PROF", [20.0, 30.0]),
            "LATITUDE": ("N_PROF", [1.0, 2.0]),
            "TIME": ("N_PROF", np.array(["2015-01-01", "2015-01-02"], dtype="datetime64[ns]")),
            "PRES": (("N_PROF", "N_LEVELS"), [[5.0, 10.0, 20.0], [5.0, 10.0, 20.0]]),
            "TEMP": (("N_PROF", "N_LEVELS"), [[12.0, 12.0, 12.0], [12.0, 12.0, 12.0]]),
        },
        coords={"N_PROF": [0, 1], "N_LEVELS": [0, 1, 2]},
        attrs={"DATA_ID": "ARGO"},
    )


def _match(ds, path, options=None, **method_params):
    return ds.argo.floatmatcher.match(
        product=dict(type="era5-hourly", source="local", selected_variables=["sst"], store=path),
        method=dict(type="nearest", **method_params),
        **(options or {}),
    )


def test_match_on_profiles(argo_profiles, grid_3d_ds, grid_file):
    out = _match(argo_profiles, grid_file(grid_3d_ds))

    # expected values: see argo_profiles
    assert out["sst_coloc"].dims == ("N_PROF",)
    npt.assert_allclose(out["sst_coloc"].values, [120.0, 231.0])
    assert out.attrs == argo_profiles.attrs


def test_match_on_points_comes_back_as_points(argo_profiles, grid_3d_ds, grid_file):
    points = argo_profiles.argo.profile2point()

    out = _match(points, grid_file(grid_3d_ds))

    assert out["sst_coloc"].dims == ("N_POINTS",)
    assert out.sizes["N_POINTS"] == points.sizes["N_POINTS"]
    # every level carries the value of its profile
    npt.assert_allclose(out["sst_coloc"].values, [120.0] * 3 + [231.0] * 3)


def test_match_on_points_k_nearest_gives_one_variable_per_neighbor(argo_profiles, grid_3d_ds,
                                                                   grid_file):
    points = argo_profiles.argo.profile2point()

    out = _match(points, grid_file(grid_3d_ds), options=dict(neigh_to_vars=True), k_nearest=2)

    # the nearest node is the one of argo_profiles, the second one is beyond radius (25 km)
    assert out["sst_coloc_s0_t0"].dims == ("N_POINTS",)
    npt.assert_allclose(out["sst_coloc_s0_t0"].values, [120.0] * 3 + [231.0] * 3)
    assert np.isnan(out["sst_coloc_s1_t0"].values).all()


def test_match_on_points_k_nearest_without_option_raises(argo_profiles, grid_3d_ds, grid_file):
    points = argo_profiles.argo.profile2point()

    with pytest.raises(ValueError, match="neigh_to_vars=True"):
        _match(points, grid_file(grid_3d_ds), k_nearest=2)


def test_match_on_points_original_shape_false_gives_profiles(argo_profiles, grid_3d_ds, grid_file):
    points = argo_profiles.argo.profile2point()

    out = _match(points, grid_file(grid_3d_ds), options=dict(original_shape=False), k_nearest=2)

    # back in profiles, with the neighbor matrix
    assert "N_POINTS" not in out.dims
    assert out["sst_coloc"].dims == ("N_PROF", "k_space", "n_time")
    npt.assert_allclose(out["sst_coloc"].values[:, 0, 0], [120.0, 231.0])
