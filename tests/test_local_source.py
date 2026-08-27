# tests/test_local_source.py
#
# resolve_path is pure (no disk) -> tested directly.
# PathTemplate needs fake files -> pytest's tmp_path, with the fake tree aligned
# on the shared `points_with_origin` fixture (dates 2015-01-01/02/03).
# The "same day -> one file" case needs points on a single day, which the shared
# fixture doesn't provide, so that one test builds a small PointSet inline.

import warnings
from pathlib import Path

import numpy as np
import pytest

from floatmatcher.local_source import LocalSource, resolve_path, ExplicitFiles, PathTemplate
from floatmatcher.pointset import PointSet


PATTERN = "{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc"


def _make_file(root: Path, y, m, d):
    """Create an empty placeholder file at the templated path."""
    p = root / f"{y}" / f"{m:02d}" / f"era5_{y}{m:02d}{d:02d}.nc"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    return p


# ───────────── resolve_path (pure, no disk) ─────────────

def test_resolve_path_substitution():
    p = resolve_path("/data", PATTERN, np.datetime64("2015-01-01"))
    assert p == "/data/2015/01/era5_20150101.nc"


def test_resolve_path_zero_padding():
    p = resolve_path("/r", "{year}{month:02d}{day:02d}", np.datetime64("2015-01-02"))
    assert p.endswith("20150102")


def test_resolve_path_is_pure_no_disk():
    p = resolve_path("/nowhere", "{year}.nc", np.datetime64("2015-01-03"))
    assert p == "/nowhere/2015.nc"


# ───────────── ExplicitFiles ─────────────

def test_explicit_single_path_becomes_list(points_with_origin):
    assert ExplicitFiles("a.nc").files_for(points_with_origin) == ["a.nc"]


def test_explicit_list_passthrough_ignores_points(points_with_origin):
    assert ExplicitFiles(["a.nc", "b.nc"]).files_for(points_with_origin) == ["a.nc", "b.nc"]


# ───────────── PathTemplate (fake tree via tmp_path) ─────────────
# points_with_origin has dates 2015-01-01 / 02 / 03.

def test_pathtemplate_resolves_present_dates(tmp_path, points_with_origin):
    for d in (1, 2, 3):
        _make_file(tmp_path, 2015, 1, d)
    out = PathTemplate(str(tmp_path), PATTERN).files_for(points_with_origin)
    assert len(out) == 3
    assert all(Path(p).exists() for p in out)


def test_pathtemplate_filters_missing_with_warning(tmp_path, points_with_origin):
    _make_file(tmp_path, 2015, 1, 1)     # day 2 missing on purpose
    _make_file(tmp_path, 2015, 1, 3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = PathTemplate(str(tmp_path), PATTERN).files_for(points_with_origin)
    assert len(out) == 2
    assert any("not found" in str(x.message) for x in w)


def test_pathtemplate_all_missing_raises(tmp_path, points_with_origin):
    with pytest.raises(FileNotFoundError):
        PathTemplate(str(tmp_path), PATTERN).files_for(points_with_origin)


def test_pathtemplate_unique_days(tmp_path):
    # several points the SAME day -> a single file. The shared fixture has three
    # distinct days, so build a small PointSet inline for this specific case.
    _make_file(tmp_path, 2015, 1, 1)
    pts = PointSet(
        lon=[0.0, 0.0, 0.0],
        lat=[0.0, 0.0, 0.0],
        time=np.array(["2015-01-01T00", "2015-01-01T06", "2015-01-01T18"],
                      dtype="datetime64[ns]"),
    )
    out = PathTemplate(str(tmp_path), PATTERN).files_for(pts)
    assert len(out) == 1


def test_from_template_builds_a_pathtemplate_source(tmp_path, points_with_origin):
    for day in (1, 2, 3):
        _make_file(tmp_path, 2015, 1, day)
    source = LocalSource.from_template(str(tmp_path), PATTERN)
    assert isinstance(source.resolver, PathTemplate)
    assert len(source.resolver.files_for(points_with_origin)) == 3


def test_from_paths_builds_an_explicitfiles_source(tmp_path, points_with_origin):
    paths = [_make_file(tmp_path, 2015, 1, 1), _make_file(tmp_path, 2015, 1, 2)]
    source = LocalSource.from_paths(paths)
    assert isinstance(source.resolver, ExplicitFiles)
    assert source.resolver.files_for(points_with_origin) == [str(p) for p in paths]


def test_explicitfiles_expands_a_directory(tmp_path, points_with_origin):
    for day in (1, 2, 3):
        _make_file(tmp_path, 2015, 1, day)
    (tmp_path / "notes.txt").write_text("ignored")
    out = ExplicitFiles(str(tmp_path)).files_for(points_with_origin)
    assert len(out) == 3
    assert out == sorted(out)
    assert all(f.endswith(".nc") for f in out)


def test_explicitfiles_mixes_files_and_directories(tmp_path, points_with_origin):
    loose = _make_file(tmp_path / "loose", 2016, 5, 9)
    tree = tmp_path / "tree"
    _make_file(tree, 2015, 1, 1)
    out = ExplicitFiles([str(loose), str(tree)]).files_for(points_with_origin)
    assert len(out) == 2
    assert str(loose) in out
