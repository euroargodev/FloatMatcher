# tests/test_sources.py

import json

import numpy as np
import pytest
import requests

from floatmatcher.mocking import FakeCDSClient
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


# ---------- CDSSource.download: cache, register and states ----------

def make_points(lon, lat, time):
    return PointSet(lon=lon, lat=lat, time=np.array(time, dtype="datetime64[ns]"))


@pytest.fixture
def p2013():
    return make_points([-30.0], [50.0], ["2013-03-18T12"])


@pytest.fixture
def p2014():
    return make_points([-30.0], [50.0], ["2014-03-18T12"])


@pytest.fixture
def p2013_wide():
    """Includes p2013: more points, so a larger area"""
    return make_points([-30.0, -40.0], [50.0, 40.0], ["2013-03-18T12", "2013-03-18T18"])


def cds_download(cache_dir, points, client, by_year=False, variables=("sst",), clear_cache=False):
    """Download the request of the points; one sub-request per year if ``by_year``

    The request is set by hand, not by build_request, so that the split does not depend on costs.
    """
    source = CDSSource(path=cache_dir, product_info=ERA5Product.product_info, client=client,
                       clear_cache=clear_cache)
    source.original_request = source._build_request(points, list(variables))
    if by_year:
        source.request = [dict(source.original_request, year=[y])
                          for y in source.original_request["year"]]
    else:
        source.request = [source.original_request]
    folder = source.download()
    return source, folder


def read_register(cache_dir):
    return json.loads((cache_dir / "register.json").read_text())


def test_download_one_request(tmp_path, p2013, fake_cds):
    source, folder = cds_download(tmp_path, p2013, fake_cds)

    assert len(fake_cds.sent) == 1
    assert sorted(f.name for f in folder.iterdir()) == ["request.json", "request_1.nc"]
    register = read_register(tmp_path)
    assert list(register) == [folder.name]
    assert register[folder.name]["state"] == "available"
    assert json.loads((folder / "request.json").read_text()) == register[folder.name]
    assert source.resolve(p2013) == [str(folder / "request_1.nc")]


def test_download_several_requests_in_one_folder(tmp_path, fake_cds):
    points = make_points([-30.0, -30.0], [50.0, 50.0], ["2013-03-18T12", "2014-03-18T12"])
    source, folder = cds_download(tmp_path, points, fake_cds, by_year=True)

    assert len(fake_cds.sent) == 2
    assert sorted(f.name for f in folder.glob("*.nc")) == ["request_1.nc", "request_2.nc"]
    register = read_register(tmp_path)
    assert len(register) == 1
    assert register[folder.name]["year"] == ["2013", "2014"]
    assert register[folder.name]["state"] == "available"
    assert len(source.resolve(points)) == 2


def test_download_same_request_skipped(tmp_path, p2013, fake_cds):
    _, first = cds_download(tmp_path, p2013, fake_cds)
    source, second = cds_download(tmp_path, p2013, fake_cds)

    assert len(fake_cds.sent) == 1
    assert second == first
    assert len(read_register(tmp_path)) == 1
    assert source.resolve(p2013) == [str(first / "request_1.nc")]


def test_download_included_request_skipped(tmp_path, p2013, p2013_wide, fake_cds):
    _, wide = cds_download(tmp_path, p2013_wide, fake_cds)
    _, folder = cds_download(tmp_path, p2013, fake_cds)

    assert len(fake_cds.sent) == 1
    assert folder == wide


def test_download_larger_request_downloaded(tmp_path, p2013, p2013_wide, fake_cds):
    _, small = cds_download(tmp_path, p2013, fake_cds)
    source, wide = cds_download(tmp_path, p2013_wide, fake_cds)

    assert len(fake_cds.sent) == 2
    assert wide != small
    # both entries kept in the register
    assert set(read_register(tmp_path)) == {small.name, wide.name}
    # the resolver reads the folder of the request only
    assert source.resolve(p2013_wide) == [str(wide / "request_2.nc")]


def test_download_extra_variable_downloaded(tmp_path, p2013, fake_cds):
    """Same area and dates, one variable more: not in the register"""
    _, first = cds_download(tmp_path, p2013, fake_cds, variables=["sst"])
    _, second = cds_download(tmp_path, p2013, fake_cds, variables=["sst", "u10"])

    assert len(fake_cds.sent) == 2
    assert second != first


def test_download_fewer_variables_skipped(tmp_path, p2013, fake_cds):
    """Same area and dates, variables among the registered ones: already downloaded"""
    _, first = cds_download(tmp_path, p2013, fake_cds, variables=["sst", "u10"])
    _, second = cds_download(tmp_path, p2013, fake_cds, variables=["sst"])

    assert len(fake_cds.sent) == 1
    assert second == first


def test_download_included_in_one_of_several(tmp_path, p2013, p2014, fake_cds):
    _, f2013 = cds_download(tmp_path, p2013, fake_cds)
    _, f2014 = cds_download(tmp_path, p2014, fake_cds)
    source, folder = cds_download(tmp_path, p2014, fake_cds)

    assert len(fake_cds.sent) == 2
    assert folder == f2014
    assert source.resolve(p2014) == [str(f2014 / "request_2.nc")]


def test_download_spread_over_several_downloaded_again(tmp_path, p2013, p2014, fake_cds):
    """Known limit: requests of the register are not combined"""
    cds_download(tmp_path, p2013, fake_cds)
    cds_download(tmp_path, p2014, fake_cds)
    both = make_points([-30.0, -30.0], [50.0, 50.0], ["2013-03-18T12", "2014-03-18T12"])
    cds_download(tmp_path, both, fake_cds, by_year=True)

    assert len(fake_cds.sent) == 4


def test_download_refused_sets_error(tmp_path, p2013):
    client = FakeCDSClient(limit=1)
    with pytest.raises(requests.HTTPError):
        cds_download(tmp_path, p2013, client)

    (entry,) = read_register(tmp_path).values()
    assert entry["state"] == "error"


def test_download_interrupted_sets_error(tmp_path, p2013, fake_cds, monkeypatch):
    def interrupted(name, request):
        raise KeyboardInterrupt
    monkeypatch.setattr(fake_cds, "retrieve", interrupted)

    with pytest.raises(KeyboardInterrupt):
        cds_download(tmp_path, p2013, fake_cds)

    (entry,) = read_register(tmp_path).values()
    assert entry["state"] == "error"


def test_download_after_error_starts_fresh(tmp_path, p2013):
    client = FakeCDSClient(limit=1)
    with pytest.raises(requests.HTTPError):
        cds_download(tmp_path, p2013, client)
    (failed,) = read_register(tmp_path)

    client.client.limit = 1000
    _, folder = cds_download(tmp_path, p2013, client)

    assert not (tmp_path / failed).exists()
    register = read_register(tmp_path)
    assert list(register) == [folder.name]
    assert register[folder.name]["state"] == "available"


def test_download_error_of_other_request_kept(tmp_path, p2013, p2014):
    client = FakeCDSClient(limit=1)
    with pytest.raises(requests.HTTPError):
        cds_download(tmp_path, p2013, client)
    (failed,) = read_register(tmp_path)

    client.client.limit = 1000
    cds_download(tmp_path, p2014, client)

    assert (tmp_path / failed).exists()
    assert read_register(tmp_path)[failed]["state"] == "error"


def test_download_pending_not_reused(tmp_path, p2013, fake_cds):
    """A pending entry (process killed during the download) is not taken as downloaded"""
    _, folder = cds_download(tmp_path, p2013, fake_cds)
    register = read_register(tmp_path)
    register[folder.name]["state"] = "pending"
    (tmp_path / "register.json").write_text(json.dumps(register))

    _, again = cds_download(tmp_path, p2013, fake_cds)

    assert len(fake_cds.sent) == 2
    assert again != folder


def test_download_zip_extracted(tmp_path):
    client = FakeCDSClient(file_format="zip")
    points = make_points([-30.0, -30.0], [50.0, 50.0], ["2013-03-18T12", "2014-03-18T12"])
    source, folder = cds_download(tmp_path, points, client, by_year=True)

    assert not list(folder.glob("*.zip"))
    assert sorted(f.name for f in folder.glob("*.nc")) == [
        "request_1_data_stream-oper_stepType-accum.nc",
        "request_1_data_stream-oper_stepType-instant.nc",
        "request_2_data_stream-oper_stepType-accum.nc",
        "request_2_data_stream-oper_stepType-instant.nc",
    ]
    assert len(source.resolve(points)) == 4


def test_download_clear_cache(tmp_path, p2013, p2014, fake_cds):
    _, old = cds_download(tmp_path, p2013, fake_cds)
    _, new = cds_download(tmp_path, p2014, fake_cds, clear_cache=True)

    assert not old.exists()
    assert list(read_register(tmp_path)) == [new.name]


def test_download_without_request_raise_error(tmp_path, fake_cds):
    source = CDSSource(path=tmp_path, product_info=ERA5Product.product_info, client=fake_cds)
    with pytest.raises(ValueError, match="No request"):
        source.download()
