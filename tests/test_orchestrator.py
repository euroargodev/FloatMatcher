# tests/test_orchestrator.py
#
# Two levels, from light to heavy:
#   1. UNIT  — Orchestrator._select_variables (pure, no I/O), tested directly.
#   2. CHAIN — the full orchestrator on a tiny fake ERA5 NetCDF in tmp_path:
#              open (LocalSource) -> normalize (Product) -> select -> GridSet -> match.

import numpy as np
import pytest
import xarray as xr

from floatmatcher.orchestrator import Orchestrator
from floatmatcher.products import ERA5Product
from floatmatcher.pointset import PointSet


# ─────────────────────────────────────────────────────────────
#  1. UNIT — _select_variables (pure, no I/O)
# ─────────────────────────────────────────────────────────────

def _grid_4vars():
    return xr.Dataset(
        {v: (("lat", "lon"), np.zeros((2, 2))) for v in ["u10", "v10", "t2m", "sst"]},
        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )


def test_select_none_keeps_all():
    out = Orchestrator._select_variables(_grid_4vars(), None)
    assert set(out.data_vars) == {"u10", "v10", "t2m", "sst"}


def test_select_single_string():
    out = Orchestrator._select_variables(_grid_4vars(), "sst")
    assert list(out.data_vars) == ["sst"]


def test_select_list_of_variables():
    out = Orchestrator._select_variables(_grid_4vars(), ["sst", "t2m"])
    assert set(out.data_vars) == {"sst", "t2m"}


def test_select_unknown_raises_and_lists_available():
    with pytest.raises(ValueError) as e:
        Orchestrator._select_variables(_grid_4vars(), "xxx")
    msg = str(e.value)
    assert "xxx" in msg            # names the bad variable
    assert "sst" in msg            # and lists what IS available


def test_select_partial_unknown_raises():
    # one valid + one invalid -> still raises (all-or-nothing)
    with pytest.raises(ValueError):
        Orchestrator._select_variables(_grid_4vars(), ["sst", "nope"])


# ─────────────────────────────────────────────────────────────
#  2. CHAIN — full orchestrator on a fake ERA5 file
# ─────────────────────────────────────────────────────────────

class DictMethod:
    """Minimal MatchupMethod stand-in: reports the variables it saw + valid mask.
    Iterates ds.data_vars exactly like the real Interpolation / NearestNeighbor."""
    def match(self, grid, points, constraints):
        ds = grid.dataset
        n = len(points.lon)

        class _R:
            pass
        r = _R()
        r.values = {v: np.zeros(n) for v in ds.data_vars}
        r.valid = np.ones(n, dtype=bool)
        r.grid_lon_range = grid.lon_range
        return r


def test_chain_all_variables_by_default(fake_era5_file, points_in_grid):
    orch = Orchestrator(ERA5Product(), DictMethod())
    res = orch.colocalize(fake_era5_file, points_in_grid)
    assert set(res.values) == {"u10", "t2m", "sst"}
    assert res.grid_lon_range == "0-360"          # native convention preserved


def test_chain_select_single_variable(fake_era5_file, points_in_grid):
    orch = Orchestrator(ERA5Product(), DictMethod())
    res = orch.colocalize(fake_era5_file, points_in_grid, variables="sst")
    assert set(res.values) == {"sst"}


def test_chain_select_multiple_variables(fake_era5_file, points_in_grid):
    orch = Orchestrator(ERA5Product(), DictMethod())
    res = orch.colocalize(fake_era5_file, points_in_grid, variables=["sst", "t2m"])
    assert set(res.values) == {"sst", "t2m"}


def test_chain_unknown_variable_raises(fake_era5_file, points_in_grid):
    orch = Orchestrator(ERA5Product(), DictMethod())
    with pytest.raises(ValueError):
        orch.colocalize(fake_era5_file, points_in_grid, variables="does_not_exist")


def test_chain_renames_valid_time_to_time(fake_era5_file, points_in_grid):
    # if valid_time -> time renaming failed, GridSet would see no 'time' coord;
    # a clean run on 3D confirms the rename happened.
    orch = Orchestrator(ERA5Product(), DictMethod())
    res = orch.colocalize(fake_era5_file, points_in_grid, variables="sst")
    assert res.valid.all()