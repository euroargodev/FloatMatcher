# tests/test_batching.py


import numpy as np
import numpy.testing as npt
import pytest

from floatmatcher.orchestrator import Orchestrator
from floatmatcher.products import ERA5Product
from floatmatcher.matchup import NearestNeighbor
from floatmatcher.pointset import PointSet
from floatmatcher.results import MatchupResult

from helpers import constraints

CONS = constraints(max_dist_km=300.0, max_time_days=5.0)


# ─────────────────────────────────────────────────────────────
#  fixtures: several ERA5-like files, SAME geometry, different time axis
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def three_era5_files(write_era5):
    """Three yearly files with the same geometry and different time axes."""
    lon = [9.0, 10.0, 11.0, 350.0]
    lat = [40.0, 30.0, 20.0]

    def encode(lo, la, ti):
        return lo * 1000 + la + ti * 0.5

    files = []
    for y in (2019, 2020, 2021):
        files.append(write_era5(
            f"era5_{y}.nc",
            times=[f"{y}-06-15T00", f"{y}-12-15T00"],
            lon=lon, lat=lat, values=encode,
        ))
    return files

@pytest.fixture
def points_across_years():
    """One point per year + a December boundary point (cross-packet merge).
    Placed close to grid nodes (the test grid is coarse) so distances stay well
    under max_dist_km and the check is about batching, not the distance cutoff."""
    return PointSet(
        lon=[10.1, 350.2, 11.0, 9.9],
        lat=[30.1, 20.1, 40.1, 30.0],
        time=np.array(["2019-06-16", "2020-12-14", "2021-06-15", "2020-12-16"],
                      dtype="datetime64[ns]"),
        origin_index=np.arange(4), origin_dim="N",
    )




# ─────────────────────────────────────────────────────────────
#  1. the key invariant: batched == single pass
# ─────────────────────────────────────────────────────────────

def test_batched_equals_single_pass(three_era5_files, points_across_years):
    """Packetizing must not change the result.
    max_files=1 forces one packet per file; max_files=100 opens all at once."""
    batched = Orchestrator(ERA5Product(), NearestNeighbor(max_files=1), CONS)\
        .colocalize(three_era5_files, points_across_years, variables="sst")
    single = Orchestrator(ERA5Product(), NearestNeighbor(max_files=100), CONS)\
        .colocalize(three_era5_files, points_across_years, variables="sst")

    npt.assert_array_equal(batched.valid, single.valid)
    npt.assert_allclose(batched.values["sst"], single.values["sst"], equal_nan=True)
    npt.assert_allclose(batched.time_delta, single.time_delta, equal_nan=True)
    npt.assert_allclose(batched.distance_km, single.distance_km, equal_nan=True)


def test_all_points_matched_across_packets(three_era5_files, points_across_years):
    """Each point finds its match in whichever packet holds its year, even when
    every file is opened as a separate packet."""
    res = Orchestrator(ERA5Product(), NearestNeighbor(max_files=1), CONS)\
        .colocalize(three_era5_files, points_across_years, variables="sst")
    assert res.valid.all()


def test_batched_values_are_correct(three_era5_files, points_across_years):
    """Not just self-consistent: the matched values are the RIGHT ones.
    Point 0 (2019-06-16, lon 10.1 -> node lon=10 idx1, lat 30.1 -> node lat=30
    idx1, June -> t=0): sst = 1*1000 + 1 + 0 = 1001."""
    res = Orchestrator(ERA5Product(), NearestNeighbor(max_files=1), CONS)\
        .colocalize(three_era5_files, points_across_years, variables="sst")
    assert res.values["sst"][0] == pytest.approx(1001.0)


# ─────────────────────────────────────────────────────────────
#  2. December/January boundary: the reason update_best exists
# ─────────────────────────────────────────────────────────────

def test_boundary_point_keeps_best_across_packets(three_era5_files):
    """A late-December point is confronted with every packet and keeps the
    smallest time_delta (nearest step is 2020-12-15, ~1 day away)."""
    pts = PointSet(lon=[10.0], lat=[30.0],
                   time=np.array(["2020-12-16"], dtype="datetime64[ns]"),
                   origin_index=np.arange(1), origin_dim="N")
    res = Orchestrator(ERA5Product(), NearestNeighbor(max_files=1), CONS)\
        .colocalize(three_era5_files, pts, variables="sst")
    assert res.valid[0]
    assert res.time_delta[0] <= 1.0 + 1e-9


# ─────────────────────────────────────────────────────────────
#  3. update_best unit tests (pure, no I/O)
# ─────────────────────────────────────────────────────────────

def _one(value, dist, delta, valid):
    return MatchupResult({"v": np.array([value])}, np.array([dist]),
                         np.array([delta]), np.array([valid]))


def test_update_best_fills_empty_accumulator():
    """~self.valid path: an empty (invalid, NaN-delta) slot is filled by any
    valid incoming match, which the delta comparison alone would miss."""
    acc = _one(np.nan, np.nan, np.nan, False)
    acc.update_best(_one(5.0, 2.0, 0.3, True))
    assert acc.valid[0]
    assert acc.values["v"][0] == 5.0
    assert acc.time_delta[0] == 0.3


def test_update_best_keeps_smaller_time_delta():
    """A temporally closer incoming match wins; all fields move together."""
    acc = _one(5.0, 2.0, 0.5, True)
    acc.update_best(_one(9.0, 1.0, 0.2, True))
    assert acc.values["v"][0] == 9.0
    assert acc.distance_km[0] == 1.0
    assert acc.time_delta[0] == 0.2


def test_update_best_ignores_worse_match():
    """Equal-or-larger delta does NOT overwrite (strictly-smaller rule -> the
    accumulator wins ties, the 'first arbitrary' choice)."""
    acc = _one(5.0, 2.0, 0.2, True)
    acc.update_best(_one(9.0, 1.0, 0.2, True))     # equal delta -> keep acc
    assert acc.values["v"][0] == 5.0
    acc.update_best(_one(9.0, 1.0, 0.9, True))     # larger delta -> keep acc
    assert acc.values["v"][0] == 5.0


def test_update_best_invalid_never_overwrites():
    """An invalid incoming match cannot clobber a valid accumulated one."""
    acc = _one(5.0, 2.0, 0.2, True)
    acc.update_best(_one(1.0, 0.0, 0.0, False))
    assert acc.values["v"][0] == 5.0
    assert acc.valid[0]


# ─────────────────────────────────────────────────────────────
#  4. geometry guard: packets must share the same lat/lon
# ─────────────────────────────────────────────────────────────

def test_mismatched_geometry_between_packets_raises(write_era5, points_across_years):
    """A different grid geometry between packets must raise."""
    lat = [40.0, 30.0, 20.0]

    def enc(lo, la, ti):
        return lo * 1000 + la + ti * 0.5

    f2019 = write_era5("g2019.nc", ["2019-06-15", "2019-12-15"],
                       [9.0, 10.0, 11.0, 350.0], lat, values=enc)
    f2020 = write_era5("g2020.nc", ["2020-06-15", "2020-12-15"],
                       [9.0, 10.0, 11.0], lat, values=enc)

    with pytest.raises(ValueError):
        Orchestrator(ERA5Product(), NearestNeighbor(max_files=1), CONS).colocalize(
            [f2019, f2020], points_across_years, variables="sst"
        )