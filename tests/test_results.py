# tests/test_results.py
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from floatmatcher.results import MatchupResult
from floatmatcher.pointset import PointSet
from floatmatcher.profile_loader import ProfileLoader


@pytest.fixture
def result():
    """ MatchupResult object - 3 profiles """
    ds = xr.Dataset(
        {"LONGITUDE": (("N_PROF",), np.array([-40.0, -30.0, -20.0])),
         "LATITUDE": (("N_PROF",), np.array([35.0, 36.0, 37.0])),
         "TIME": (("N_PROF",), np.array(["2018-01-03", "2018-01-01", "2018-01-02"],
                                        dtype="datetime64[ns]"))},
        coords={"N_PROF": np.array([10, 11, 12])},
    )
    points = ProfileLoader.from_xrdataset(ds)
    return MatchupResult(
        values={"dummy_variable": np.array([280.0, 281.0, np.nan])},
        distance_km=np.array([1.0, 2.0, np.nan]),
        time_delta=np.array([0.1, 0.2, np.nan]),
        valid=np.array([True, True, False]),
        points=points,
    )

# @pytest.fixture
# def argo_like_ds():
#     return xr.Dataset(
#         {"LONGITUDE": (("N_PROF",), np.array([-40.0, -30.0, -20.0])),
#          "LATITUDE": (("N_PROF",), np.array([35.0, 36.0, 37.0])),
#          "TIME": (("N_PROF",), np.array(["2018-01-03", "2018-01-01", "2018-01-02"],
#                                         dtype="datetime64[ns]"))},
#         coords={"N_PROF": np.array([10, 11, 12])},
#     )


# ---------- to_dataset ----------

def test_to_dataset_adds_coloc_variable(result):
    out = result.to_dataset()
    assert out["dummy_variable_coloc"].dims == ("N_PROF",)
    assert out["dummy_variable_coloc"].sel(N_PROF=10) == 280.0


def test_to_dataset_does_not_mutate_the_source(result):
    result.to_dataset()
    assert "dummy_variable_coloc" not in result.points.origin_ds.data_vars


def test_to_dataset_without_provenance_raises():
    """Raw-array points carry no dataset, so they cannot be reinjected."""
    points = PointSet([-45.0, -44.0, -43.0], [32.0, 33.0, 34.0],
                      np.array(["2015-01-01", "2015-01-02", "2015-01-03"],
                               dtype="datetime64[ns]"))
    res = MatchupResult(values={"dummy_variable": np.zeros(3)},
                        distance_km=np.zeros(3), time_delta=np.zeros(3),
                        valid=np.ones(3, bool), points=points)
    with pytest.raises(ValueError):
        res.to_dataset()
