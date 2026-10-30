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
def cds_source(fake_cds):
    return CDSSource(path="/cache", product_info=ERA5Product.product_info, client=fake_cds)


@pytest.fixture
def points():
    return PointSet(lon=[-45.0, -44.0], lat=[32.0, 33.0],
                    time=np.array(["2015-01-01", "2015-01-02"], dtype="datetime64[ns]"))


def test_cds_request_uses_cds_variable_names(cds_source, points):
    requests = cds_source.build_request(points, ["sst", "u10"])
    assert requests[0]["variable"] == ["sea_surface_temperature", "10m_u_component_of_wind"]


def test_cds_request_unknown_variable_raise_error(cds_source, points):
    with pytest.raises(ValueError, match="unknown variables"):
        cds_source.build_request(points, ["sea_surface_temperature"])


def test_cds_request_without_variables_raise_error(cds_source, points):
    with pytest.raises(ValueError, match="needs a list of variables"):
        cds_source.build_request(points, None)


def test_local_source_has_no_request(points):
    with pytest.raises(ValueError):
        LocalSource(path="/data").build_request(points, ["sst"])


def test_cds_source_uses_given_client(fake_cds):
    source = CDSSource(path="/cache", product_info=ERA5Product.product_info, client=fake_cds)
    assert source.client is fake_cds


def test_cds_options_default_to_all_values(fake_cds):
    source = CDSSource(path="/cache", product_info=ERA5Product.product_info, client=fake_cds)
    assert source._options_request({}) == {"time": [f"{h:02d}:00" for h in range(24)]}


def test_cds_options_subset_of_hours(fake_cds):
    source = CDSSource(path="/cache", product_info=ERA5Product.product_info, client=fake_cds)
    assert source._options_request({"time": ["02:00", "14:00"]}) == {"time": ["02:00", "14:00"]}


def test_cds_options_unknown_option_raise_error(fake_cds):
    source = CDSSource(path="/cache", product_info=ERA5Product.product_info, client=fake_cds)
    with pytest.raises(ValueError, match="unknown options"):
        source._options_request({"hours": ["02:00"]})


def test_cds_options_unavailable_value_raise_error(fake_cds):
    source = CDSSource(path="/cache", product_info=ERA5Product.product_info, client=fake_cds)
    with pytest.raises(ValueError, match="not available"):
        source._options_request({"time": ["02:30"]})
    with pytest.raises(ValueError, match="empty list"):
        source._options_request({"time": []})
