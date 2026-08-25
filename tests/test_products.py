# tests/test_products.py

import numpy as np
import pytest
import xarray as xr

from floatmatcher.products import (
    rename_coords,
    to_standard,
    ERA5Product,
    LUTProduct,
)


def _era5_raw():
    """Raw ERA5-like dataset with source names longitude/latitude/time."""
    return xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), np.zeros((2, 3, 3)))},
        coords={
            "time": np.array(["2015-01-01", "2015-01-02"], dtype="datetime64[ns]"),
            "latitude": [0.0, 1.0, 2.0],
            "longitude": [10.0, 11.0, 12.0],
        },
    )


# ───────────── rename_coords ─────────────

def test_rename_maps_present_keys():
    out = rename_coords(_era5_raw(), {"longitude": "lon", "latitude": "lat", "time": "time"})
    assert "lon" in out.coords and "lat" in out.coords


def test_rename_ignores_absent_keys():
    # 'foo' does not exist -> skipped, not an error (tolerant policy)
    out = rename_coords(_era5_raw(), {"longitude": "lon", "foo": "bar"})
    assert "lon" in out.coords
    assert "bar" not in out.variables


def test_rename_identity_mapping_is_noop():
    # LUT case: source == target must pass cleanly
    ds = xr.Dataset({"chl": (("lat", "lon"), np.zeros((2, 2)))},
                    coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})
    out = rename_coords(ds, {"lon": "lon", "lat": "lat"})
    assert "lon" in out.coords and "lat" in out.coords


def test_rename_bare_dimension():
    # a dimension without a coordinate can still be renamed
    ds = xr.Dataset({"v": (("obs",), [1.0, 2.0, 3.0])})
    out = rename_coords(ds, {"obs": "points"})
    assert "points" in out.dims and "obs" not in out.dims


def test_rename_does_not_mutate_input():
    ds = _era5_raw()
    _ = rename_coords(ds, {"longitude": "lon"})
    assert "longitude" in ds.coords     # original left untouched


# ───────────── to_standard ─────────────

def test_to_standard_renames_end_to_end():
    out = to_standard(_era5_raw(), ERA5Product.COORD_MAP)
    assert {"lat", "lon", "time"} <= set(out.coords)


# ───────────── Products ─────────────

def test_era5_normalize():
    out = ERA5Product().normalize(_era5_raw())
    assert "lon" in out.coords and "lat" in out.coords


def test_era5_declared_attributes():
    assert ERA5Product.LON_RANGE == "0-360"
    assert ERA5Product.COORD_MAP["longitude"] == "lon"


def test_lut_normalize():
    ds = xr.Dataset({"chl": (("lat", "lon"), np.zeros((2, 2)))},
                    coords={"lat": [0.0, 1.0], "lon": [-10.0, -9.0]})
    out = LUTProduct().normalize(ds)
    assert "lon" in out.coords and "lat" in out.coords


def test_lut_declared_attributes():
    assert LUTProduct.LON_RANGE == "-180-180"