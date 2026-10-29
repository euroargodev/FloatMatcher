# tests/test_sources.py

import numpy as np
import pytest

from floatmatcher.pointset import PointSet
from floatmatcher.product import ERA5Product
from floatmatcher.resolver import PathTemplate, ExplicitFiles
from floatmatcher.sources import LocalSource, CDSSource


# ---------- LocalSource ----------

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


# ---------- CDSSource ----------

@pytest.fixture
def cds_source():
    return CDSSource(path="/cache", product_info=ERA5Product.product_info)


@pytest.fixture
def points():
    return PointSet(lon=[-45.0, -44.0], lat=[32.0, 33.0],
                    time=np.array(["2015-01-01", "2015-01-02"], dtype="datetime64[ns]"))


def test_cds_request_uses_cds_variable_names(cds_source, points):
    request = cds_source.build_request(points, ["sst", "u10"])
    assert request["variable"] == ["sea_surface_temperature", "10m_u_component_of_wind"]


def test_cds_request_unknown_variable_raise_error(cds_source, points):
    with pytest.raises(ValueError, match="unknown variables"):
        cds_source.build_request(points, ["sea_surface_temperature"])


def test_cds_request_without_variables_raise_error(cds_source, points):
    with pytest.raises(ValueError, match="needs a list of variables"):
        cds_source.build_request(points, None)


def test_local_source_has_no_request(points):
    with pytest.raises(ValueError):
        LocalSource(path="/data").build_request(points, ["sst"])
