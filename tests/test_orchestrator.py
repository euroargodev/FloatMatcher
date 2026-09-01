# tests/test_orchestrator.py
#
# The orchestrator on a tiny ERA5-like NetCDF written in tmp_path:
# resolve -> open -> normalize -> select -> GridSet -> nearest -> MatchupResult.

import numpy as np
import numpy.testing as npt
import pytest

from floatmatcher.matchup import NearestNeighbor
from floatmatcher.orchestrator import Orchestrator
from floatmatcher.products import ERA5Product
from floatmatcher.profile_loader import ProfileLoader


def test_match_nearest_get_good_node(era5_files):
    points = ProfileLoader.from_arrays(
        lon=[11.1, 9.05, 349.9],
        lat=[30.05, 39.95, 20.1],
        time=np.array(["2015-06-01T02", "2015-06-02T00", "2015-06-02T22"], dtype="datetime64[ns]"),
    )
    product = ERA5Product.from_local(path=era5_files)

    res = Orchestrator(points, "sst", product).match(NearestNeighbor())

    assert res.valid.all()
    assert set(res.values) == {"sst"} # only sst has been taken into account
    npt.assert_allclose(res.values["sst"], [121.03, 1.23, 232.23], atol=1e-4)
    assert (res.distance_km < 25).all() 
    assert (res.time_delta < 5 * 3600).all()          # seconds - 5h
    assert len(np.unique(res.distance_km)) == 3


def test_match_nearest_rejects_out_of_range_points(era5_files):
    points = ProfileLoader.from_arrays(
        lon=[11.1,    230, 11.1],
        lat=[30.05, -40.0, 30.05],
        time=np.array(["2015-06-01T02",      # p0 in range
                       "2015-06-01T02",      # p1 far away in space
                       "2015-09-01T02"],     # p2 far away in time
                      dtype="datetime64[ns]"),
    )
    product = ERA5Product.from_local(path=era5_files)
    orch = Orchestrator(points, "sst", product)

    res = orch.match(NearestNeighbor())

    assert res.valid.tolist() == [True, False, False]
    npt.assert_allclose(res.values["sst"][:1], [121.03], atol=1e-4)
    assert np.isnan(res.values["sst"][1:]).all()      # no value for the rejected
    assert np.isnan(res.distance_km[1:]).all()        # nor distance
    assert np.isnan(res.time_delta[1:]).all()         # nor time gap

    # p1 was out on DISTANCE only
    loose_dist = orch.match(NearestNeighbor(max_dist_km=100000))
    assert loose_dist.valid.tolist() == [True, True, False]

    # p2 was out on TIME only
    loose_time = orch.match(NearestNeighbor(max_time=np.timedelta64(300, "D")))
    assert loose_time.valid.tolist() == [True, False, True]


def test_match_nearest_over_two_variable(era5_files):
    points = ProfileLoader.from_arrays(
        lon=[11.1], lat=[30.05],
        time=np.array(["2015-06-01T02"], dtype="datetime64[ns]"),
    )
    product = ERA5Product.from_local(path=era5_files)

    res = Orchestrator(points, ["sst", "t2m"], product).match(NearestNeighbor())

    assert set(res.values) == {"sst", "t2m"}
    npt.assert_allclose(res.values["sst"], [121.03], atol=1e-4)
    npt.assert_allclose(res.values["t2m"], [1121.03], atol=1e-4)
