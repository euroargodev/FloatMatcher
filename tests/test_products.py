# tests/test_products.py

import numpy as np
import numpy.testing as npt
import xarray as xr
import pytest

from floatmatcher.resolver import PathTemplate, ExplicitFiles
from floatmatcher.products import (
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


# ───────────── to_standard ─────────────

def test_rename_maps_present_keys():
    out = to_standard(_era5_raw(), {"longitude": "lon", "latitude": "lat", "valid_time": "time"})
    assert "lon" in out.coords and "lat" in out.coords and "time" in out.coords


def test_rename_ignores_absent_keys():
    # 'foo' does not exist -> skipped, not an error
    out = to_standard(_era5_raw(), {"longitude": "lon", "foo": "bar"})
    assert "lon" in out.coords
    assert "bar" not in out.variables


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
    out = to_standard(ds, {"longitude": "lon"})
    assert "longitude" in ds.coords     # original left untouched



# ───────────── Products ─────────────

def test_era5_normalize():
    out = ERA5Product().normalize(_era5_raw())
    assert "lon" in out.coords and "lat" in out.coords and "time" in out.coords


def test_lut_normalize():
    ds = xr.Dataset({"chl": (("lat", "lon"), np.zeros((2, 2)))},
                    coords={"lat": [0.0, 1.0], "lon": [-10.0, -9.0]})
    out = LUTProduct().normalize(ds)
    assert "lon" in out.coords and "lat" in out.coords


def test_from_local_with_pattern_builds_a_pathtemplate():
    product = ERA5Product.from_local(path="/data", pattern="{year}/x.nc")
    assert isinstance(product.resolver, PathTemplate)


def test_from_local_without_pattern_builds_explicitfiles():
    product = ERA5Product.from_local(path="/data/2018")
    assert isinstance(product.resolver, ExplicitFiles)

    
def test_from_local_without_path_raise_error():
    product = ERA5Product.from_local(pattern="{year}/x.nc")
    assert isinstance(product.resolver, PathTemplate)


def test_from_local_returns_the_good_product():
    assert isinstance(ERA5Product.from_local(path="/data"), ERA5Product)
    assert isinstance(LUTProduct.from_local(path="/data"), LUTProduct)


def test_from_local_without_path_raise_error():
    with pytest.raises(ValueError):
        ERA5Product.from_local(pattern="{year}/x.nc")     # pattern sans path
    with pytest.raises(ValueError):
        ERA5Product.from_local() 