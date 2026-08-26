# tests/test_end_to_end.py
#
# End-to-end: PathTemplate -> Orchestrator -> batched nearest. 

import numpy as np
import pytest
from pathlib import Path

from floatmatcher.local_source import PathTemplate, LocalSource, resolve_path
from floatmatcher.orchestrator import Orchestrator
from floatmatcher.products import ERA5Product
from floatmatcher.nearest import NearestNeighbor
from floatmatcher.pointset import PointSet

from helpers import constraints

CONS = constraints(max_dist_km=800.0, max_time_days=1.0)


# data shape: 2018/10/era5_single-levels_20181008.nc
PATTERN = "{year}/{month:02d}/era5_single-levels_{year}{month:02d}{day:02d}.nc"

LAT = np.array([40.0, 30.0, 20.0])           # decreasing (ERA5)
LON = np.array([9.0, 10.0, 11.0, 350.0])     # 0-360 (a lon > 180)


def _day_field(lo, la, hh):
    """Encode node+hour in the value, so the matched value is predictable."""
    return lo * 1000 + la + hh * 0.1


@pytest.fixture
def era5_tree(tmp_path, write_era5):
    """A small ERA5 tree spanning two months, plus the root path."""
    root = str(tmp_path / "fm_data" / "era5_daily")
    for (y, m, d) in [(2018, 10, 8), (2018, 10, 9), (2018, 11, 3)]:
        date = np.datetime64(f"{y}-{m:02d}-{d:02d}")
        name = Path(resolve_path(root, PATTERN, date)).relative_to(tmp_path)
        stamps = []
        for h in range(0, 24, 6):
            stamps.append(f"{y}-{m:02d}-{d:02d}T{h:02d}")
        vt = np.array(stamps, dtype="datetime64[ns]")
        write_era5(str(name), times=vt, lon=LON, lat=LAT, values=_day_field)
    return root


@pytest.fixture
def points():
    """Points across the tree's days (two share the Oct-08 file)."""
    return PointSet(
        lon=[9.5, 10.5, 9.8, 10.2],
        lat=[35.0, 25.0, 35.0, 32.0],
        time=np.array(["2018-10-08T03", "2018-10-08T21", "2018-10-09T09",
                       "2018-11-03T14"], dtype="datetime64[ns]"),
        origin_index=np.arange(4), origin_dim="N",
    )


def test_pathtemplate_resolves_only_needed_days(era5_tree, points):
    """The resolver returns one file per unique day present in the points."""
    files = PathTemplate(era5_tree, PATTERN).files_for(points)
    assert len(files) == 3                      # Oct-08, Oct-09, Nov-03
    assert all(Path(f).exists() for f in files)


def test_pathtemplate_through_batched_nearest(era5_tree, points):
    """PathTemplate feeding the batched nearest loop colocalizes every point."""
    source = LocalSource(PathTemplate(era5_tree, PATTERN))
    orch = Orchestrator(ERA5Product(), NearestNeighbor(max_files=1), CONS)
    res = orch.colocalize(source, points, variables="sst")
    assert res.valid.all()                      # all points matched
    assert len(res.values["sst"]) == 4


def test_pathtemplate_matches_explicitfiles_fallback(era5_tree, points):
    """The two entry paths agree: a LocalSource(PathTemplate) and passing the
    resolved paths directly (wrapped in ExplicitFiles by the orchestrator)."""
    files = PathTemplate(era5_tree, PATTERN).files_for(points)

    via_resolver = Orchestrator(ERA5Product(), NearestNeighbor(max_files=1), CONS)\
        .colocalize(LocalSource(PathTemplate(era5_tree, PATTERN)), points, variables="sst")
    via_paths = Orchestrator(ERA5Product(), NearestNeighbor(max_files=1), CONS)\
        .colocalize(files, points, variables="sst")     # bare list -> ExplicitFiles

    np.testing.assert_array_equal(via_resolver.valid, via_paths.valid)
    np.testing.assert_allclose(via_resolver.values["sst"], via_paths.values["sst"],
                               equal_nan=True)


def test_pathtemplate_batched_equals_single(era5_tree, points):
    """Packetizing the resolved files does not change the result."""
    def src():
        return LocalSource(PathTemplate(era5_tree, PATTERN))

    batched = Orchestrator(ERA5Product(), NearestNeighbor(max_files=1), CONS)\
        .colocalize(src(), points, variables="sst")
    single = Orchestrator(ERA5Product(), NearestNeighbor(max_files=100), CONS)\
        .colocalize(src(), points, variables="sst")
    np.testing.assert_allclose(batched.values["sst"], single.values["sst"],
                               equal_nan=True)