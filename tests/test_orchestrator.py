# tests/test_orchestrator.py
#
# Two levels, from light to heavy:
#   1. UNIT  — _select_variables (pure, no I/O), tested directly.
#   2. CHAIN — the full orchestrator on a tiny fake ERA5 NetCDF in tmp_path:
#              open (LocalSource) -> normalize (Product) -> select -> GridSet -> match.

import numpy as np
import pytest
import xarray as xr

from floatmatcher.utils import _select_variables


# ─────────────────────────────────────────────────────────────
#  1. UNIT — _select_variables (pure, no I/O)
# ─────────────────────────────────────────────────────────────

def _grid_4vars():
    return xr.Dataset(
        {v: (("lat", "lon"), np.zeros((2, 2))) for v in ["u10", "v10", "t2m", "sst"]},
        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )


def test_select_none_keeps_all():
    out = _select_variables(_grid_4vars(), None)
    assert set(out.data_vars) == {"u10", "v10", "t2m", "sst"}


def test_select_single_string():
    out = _select_variables(_grid_4vars(), "sst")
    assert list(out.data_vars) == ["sst"]


def test_select_list_of_variables():
    out = _select_variables(_grid_4vars(), ["sst", "t2m"])
    assert set(out.data_vars) == {"sst", "t2m"}


def test_select_unknown_raises_and_lists_available():
    with pytest.raises(ValueError) as e:
        _select_variables(_grid_4vars(), "xxx")
    msg = str(e.value)
    assert "xxx" in msg            # names the bad variable
    assert "sst" in msg            # and lists what IS available


def test_select_partial_unknown_raises():
    # one valid + one invalid -> still raises (all-or-nothing)
    with pytest.raises(ValueError):
        _select_variables(_grid_4vars(), ["sst", "nope"])

