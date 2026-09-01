# tests/test_resolver.py



import warnings
from pathlib import Path

import numpy as np
import pytest

from floatmatcher.resolver import resolve_path, ExplicitFiles, PathTemplate
from floatmatcher.pointset import PointSet


PATTERN = "{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc"


def _make_file(root: Path, y, m, d):
    """Create an empty file at the templated path."""
    p = root / f"{y}" / f"{m:02d}" / f"era5_{y}{m:02d}{d:02d}.nc"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    return p


def _pts(*timestamps):
    """A PointSet at those instants; positions are irrelevant here."""
    n = len(timestamps)
    return PointSet(lon=[0.0] * n, lat=[0.0] * n,
                    time=np.array(timestamps, dtype="datetime64[ns]"))


# ───────────── resolve_path (pure, no disk) ─────────────

def test_resolve_path():
    p = resolve_path("/data", PATTERN, np.datetime64("2015-01-01"))
    assert p == "/data/2015/01/era5_20150101.nc"


# ───────────── ExplicitFiles ─────────────

def test_explicit_single_path_becomes_list():
    assert ExplicitFiles("a.nc").files_for() == ["a.nc"]


def test_explicit_list_is_returned_directly():
    assert ExplicitFiles(["a.nc", "b.nc"]).files_for() == ["a.nc", "b.nc"]


def test_explicitfiles_expands_a_directory(tmp_path):
    for day in (1, 2, 3):                   # creates 3 files .nc with good pattern
        _make_file(tmp_path, 2015, 1, day)
    (tmp_path / "notes.txt").write_text("ignored") # add a .txt in files list
    out = ExplicitFiles(str(tmp_path)).files_for()
    assert len(out) == 3                        # only 3 files matching .nc found
    assert out == sorted(out)                   # list already sorted
    assert [Path(f).name for f in out] == ["era5_20150101.nc",
                                          "era5_20150102.nc",
                                          "era5_20150103.nc"]


def test_explicitfiles_mixes_files_and_directories(tmp_path):
    """A named file is kept as is, a directory is expanded, and both land in
    the same list -- in the order the caller gave them."""
    loose = _make_file(tmp_path / "loose", 2016, 5, 9)
    tree = tmp_path / "tree"
    inner = _make_file(tree, 2015, 1, 1)

    out = ExplicitFiles([str(loose), str(tree)]).files_for()

    assert out == [str(loose), str(inner)]


# ───────────── PathTemplate (fake tree via tmp_path) ─────────────
# points_object has dates 2015-01-01 / 02 / 03.

def test_pathtemplate_resolves_present_dates(tmp_path, points_object):
    for d in (1, 2, 3):
        _make_file(tmp_path, 2015, 1, d)
    out = PathTemplate(str(tmp_path), PATTERN).files_for(points_object)
    assert len(out) == 3
    for p in out : 
        assert Path(p).exists()


def test_pathtemplate_filters_missing_with_warning(tmp_path, points_object):
    _make_file(tmp_path, 2015, 1, 1)     # day 2 missing on purpose
    _make_file(tmp_path, 2015, 1, 3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PathTemplate(str(tmp_path), PATTERN).files_for(points_object)
    assert len(w) == 1 # only 1 warning "not found"
    assert "era5_20150102.nc" in str(w[0].message) # explicit what is missig aka the 02/01/2015


def test_pathtemplate_all_missing_raises(tmp_path, points_object):
    with pytest.raises(FileNotFoundError):
        PathTemplate(str(tmp_path), PATTERN).files_for(points_object)


def test_same_day_points_give_one_file_and_no_warning(tmp_path):
    """ several points the SAME day -> a single file"""
    _make_file(tmp_path, 2015, 1, 1) # neighbors doesnt exists
    pts = _pts("2015-01-01T00", "2015-01-01T06", "2015-01-01T18")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = PathTemplate(str(tmp_path), PATTERN, pad=1).files_for(pts)
    assert len(w) == 0      # no warning found even if neighbors are missing
    assert [Path(f).name for f in out] == ["era5_20150101.nc"]   # good file has beed found


# ───────────── padding ─────────────

def test_pad_pulls_the_neighbouring_files(tmp_path):
    """A point in the day may match in the next or previous file --> returned both"""
    for day in (1, 2, 3):
        _make_file(tmp_path, 2015, 1, day)

    out = PathTemplate(str(tmp_path), PATTERN, pad=1).files_for(_pts("2015-01-02"))

    assert [Path(f).name for f in out] == ["era5_20150101.nc",
                                           "era5_20150102.nc",
                                           "era5_20150103.nc"]


def test_pad_zero_keeps_only_the_point_dates(tmp_path):
    for day in (1, 2, 3):
        _make_file(tmp_path, 2015, 1, day)
    out = PathTemplate(str(tmp_path), PATTERN, pad=0).files_for(_pts("2015-01-02T23"))
    assert [Path(f).name for f in out] == ["era5_20150102.nc"]
