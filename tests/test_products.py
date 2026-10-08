# tests/test_products.py

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from floatmatcher.product import (
    to_standard,
    ERA5Product,
    LUTProduct,
)


def _era5_raw():
    """Raw ERA5-like dataset with source names longitude/latitude/time."""
    return xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), np.zeros((2, 3, 3)))},
        coords={
            "valid_time": np.array(["2015-01-01", "2015-01-02"], dtype="datetime64[ns]"),
            "latitude": [0.0, 1.0, 2.0],
            "longitude": [10.0, 11.0, 12.0],
        },
    )


# ---------- to_standard ----------

def test_rename_maps_present_keys():
    out = to_standard(_era5_raw(), {"longitude": "lon", "latitude": "lat", "valid_time": "time"})
    assert "lon" in out.coords and "lat" in out.coords and "time" in out.coords


def test_rename_ignores_absent_keys():
    out = to_standard(_era5_raw(), {"longitude": "lon", "foo": "bar"})
    assert "lon" in out.coords
    assert "bar" not in out.variables  # 'foo' does not exist -> skipped, not an error


def test_rename_mapping_already_well_named():
    ds = xr.Dataset({"chl": (("lat", "lon"), np.zeros((2, 2)))},
                    coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})
    out = to_standard(ds, {"lon": "lon", "lat": "lat"})
    assert "lon" in out.coords and "lat" in out.coords


def test_rename_bare_dimension():
    # a dimension without a coordinate can still be renamed
    ds = xr.Dataset({"v": (("obs",), [1.0, 2.0, 3.0])})
    out = to_standard(ds, {"obs": "points"})
    assert "points" in out.dims and "obs" not in out.dims


def test_data_variable_is_promoted_to_coord():
    """lon/lat carried as data variables (anonymous dims) become coordinates:
    GridSet reads them from ds.coords, never from ds.data_vars."""
    ds = xr.Dataset(
        {
            "sst": (("y", "x"), np.zeros((2, 2))),
            "latitude": (("y",), np.array([0.0, 1.0])),
            "longitude": (("x",), np.array([10.0, 11.0])),
        },
    )
    assert "latitude" in ds.data_vars                    # not a coord at this point

    out = to_standard(ds, {"longitude": "lon", "latitude": "lat"})

    assert "lon" in out.coords and "lat" in out.coords
    assert "lon" not in out.data_vars and "lat" not in out.data_vars
    npt.assert_allclose(out["lon"].values, [10.0, 11.0])


def test_rename_does_not_affect_input():
    ds = _era5_raw()
    _ = to_standard(ds, {"longitude": "lon"})
    assert "longitude" in ds.coords     # original left untouched



# ---------- Products ----------

def test_era5_normalize(tmp_path):
    out = ERA5Product("local", tmp_path, None).normalize(_era5_raw())
    assert "lon" in out.coords and "lat" in out.coords and "time" in out.coords


def test_lut_normalize(tmp_path):
    ds = xr.Dataset({"chl": (("lat", "lon"), np.zeros((2, 2)))},
                    coords={"lat": [0.0, 1.0], "lon": [-10.0, -9.0]})
    out = LUTProduct("local", tmp_path, None).normalize(ds)
    assert "lon" in out.coords and "lat" in out.coords


def test_product_built_from_local_keeps_its_path():
    assert ERA5Product("local", "/data", None).path == "/data"
    assert LUTProduct("local", "/data", None).path == "/data"


# ---------- open : grid validation ----------
# grids come from conftest (grid_3d_ds / grid_2d_ds), opened through open_product

def test_3d_grid_keeps_its_time_axis(open_product, grid_3d_ds):
    """A grid with a time coord is opened as it is."""
    assert "time" in open_product(grid_3d_ds).src_dataset.coords


def test_2d_grid_has_no_time_axis(open_product, grid_2d_ds):
    """A grid without a time coord is opened as it is."""
    assert "time" not in open_product(grid_2d_ds, LUTProduct).src_dataset.coords


def test_src_dataset_keeps_the_grid(open_product, grid_3d_ds):
    """open keeps the whole dataset, the variable selection is done afterwards."""
    product = open_product(grid_3d_ds)
    xr.testing.assert_equal(product.src_dataset, grid_3d_ds)


def test_missing_lat_3d_raises(open_product, grid_3d_ds):
    """A grid without lat is rejected."""
    with pytest.raises(ValueError):
        open_product(grid_3d_ds.drop_vars("lat"))


def test_missing_lat_2d_raises(open_product, grid_2d_ds):
    """A grid without lat is rejected."""
    with pytest.raises(ValueError):
        open_product(grid_2d_ds.drop_vars("lat"), LUTProduct)


def test_no_data_variable_raises(open_product, grid_2d_ds):
    """A grid with coords but no data variable is rejected."""
    with pytest.raises(ValueError):
        open_product(grid_2d_ds.drop_vars("v"), LUTProduct)   # removes the only variable


# ---------- open : shuffled and duplicated coordinates ----------

def test_shuffled_lon_is_accepted(open_product, grid_3d_ds):
    """ order does not matter: the KDTree works on a point cloud"""
    ds = grid_3d_ds.isel(lon=[2, 0, 1])            # [30,10,20]
    assert open_product(ds).src_dataset.sizes["lon"] == 3


def test_duplicate_lon_raises(open_product, grid_3d_ds):
    """two nodes at the same position would make the nearest lookup ambiguous"""
    ds = grid_3d_ds.isel(lon=[0, 1, 1])
    with pytest.raises(ValueError):
        open_product(ds)


def test_duplicate_time_raises(open_product, grid_3d_ds):
    """the same timestamp twice (overlapping files) would make the temporal lookup ambiguous"""
    ds = grid_3d_ds.isel(time=[0, 0])
    with pytest.raises(ValueError, match="duplicate timestamps"):
        open_product(ds)


def test_decreasing_time_is_accepted(open_product, grid_3d_ds):
    """ decreasing time is accepted """
    ds = grid_3d_ds.isel(time=slice(None, None, -1))
    assert open_product(ds).src_dataset.sizes["time"] == 2
