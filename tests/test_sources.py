# tests/test_sources.py

import pytest

from floatmatcher.resolver import PathTemplate, ExplicitFiles
from floatmatcher.sources import LocalSource


# ───────────── LocalSource ─────────────

def test_local_with_pattern_builds_a_pathtemplate():
    source = LocalSource(path="/data", pattern="{year}/x.nc")
    assert isinstance(source._resolver, PathTemplate)


def test_local_without_pattern_builds_explicitfiles():
    source = LocalSource(path="/data/2018")
    assert isinstance(source._resolver, ExplicitFiles)


def test_local_without_path_raise_error():
    with pytest.raises(ValueError):
        LocalSource(path="", pattern="{year}/x.nc")     # pattern sans path
    with pytest.raises(ValueError):
        LocalSource(path="")
