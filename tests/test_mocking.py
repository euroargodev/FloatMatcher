# tests/test_mocking.py: FakeCDSClient answers like CDS, with a toy cost

import pytest
import requests

from floatmatcher.mocking import FakeCDSClient

DATASET = "reanalysis-era5-single-levels"


def make_request(n_days, n_hours, n_variables=1):
    return {
        "variable": [f"v{i}" for i in range(n_variables)],
        "year": ["2003"],
        "month": ["01"],
        "day": [f"{d:02d}" for d in range(1, n_days + 1)],
        "time": [f"{h:02d}:00" for h in range(n_hours)],
    }


def test_estimate_costs_toy_cost(fake_cds):
    estimate = fake_cds.client.estimate_costs(DATASET, make_request(10, 4, 2))
    assert estimate == {"id": "size", "cost": 80.0, "limit": 1000.0}


def test_retrieve_too_large_raise_403():
    fake_cds = FakeCDSClient(limit=79)
    with pytest.raises(requests.HTTPError, match="cost limits exceeded") as error:
        fake_cds.retrieve(DATASET, make_request(10, 4, 2))
    assert error.value.response.status_code == 403
    assert len(fake_cds.sent) == 1


def test_download_writes_file_at_target(fake_cds, tmp_path):
    request = make_request(10, 4)
    target = fake_cds.retrieve(DATASET, request).download(tmp_path / "out.nc")
    assert target == str(tmp_path / "out.nc")
    assert (tmp_path / "out.nc").exists()
    assert fake_cds.sent == [request]
