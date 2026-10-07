# tests/test_results.py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from floatmatcher.matchup_results import MatchupResult
from floatmatcher.pointset import PointSet


@pytest.fixture
def ds_result():
    """ MatchupResult object - 3 profiles """
    ds = xr.Dataset(
        {"LONGITUDE": (("N_PROF",), np.array([-40.0, -30.0, -20.0])),
         "LATITUDE": (("N_PROF",), np.array([35.0, 36.0, 37.0])),
         "TIME": (("N_PROF",), np.array(["2018-01-03", "2018-01-01", "2018-01-02"],
                                        dtype="datetime64[ns]"))},
        coords={"N_PROF": np.array([10, 11, 12])},
    )
    points = PointSet(
        lon=ds["LONGITUDE"].values,
        lat=ds["LATITUDE"].values,
        time=ds["TIME"].values,
        points_dim="N_PROF",
        original_data=ds,
    )
    return MatchupResult(
        values={"dummy_variable": np.array([280.0, 281.0, np.nan])},
        distance_km=np.array([1.0, 2.0, np.nan]),
        time_delta=np.array([0.1, 0.2, np.nan]),
        valid=np.array([True, True, False]),
        points=points,
    )

@pytest.fixture
def df_result():
    """ MatchupResult object - 3 profiles from a DataFrame """
    df = pd.DataFrame(
        {"longitude": [-40.0, -30.0, -20.0],
         "latitude": [35.0, 36.0, 37.0],
         "date": pd.to_datetime(["2018-01-03", "2018-01-01", "2018-01-02"])},
        index=pd.Index([10, 11, 12], name="profile"),
    )
    return MatchupResult(
        values={"dummy_variable": np.array([280.0, 281.0, np.nan])},
        distance_km=np.array([1.0, 2.0, np.nan]),
        time_delta=np.array([0.1, 0.2, np.nan]),
        valid=np.array([True, True, False]),
        points=PointSet.from_dataframe(df),
    )

@pytest.fixture
def raw_result():
    """Raw-array points carry no origin, so they cannot be reinjected."""
    points = PointSet(np.array([-45.0, -44.0, -43.0]), np.array([32.0, 33.0, 34.0]),
                      np.array(["2015-01-01", "2015-01-02", "2015-01-03"],
                               dtype="datetime64[ns]"))
    return MatchupResult(values={"dummy_variable": np.zeros(3)},
                         distance_km=np.zeros(3), time_delta=np.zeros(3),
                         valid=np.ones(3, bool), points=points)

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

def test_to_dataset_adds_coloc_variable(ds_result):
    out = ds_result.to_dataset()
    assert out["dummy_variable_coloc"].dims == ("N_PROF",)
    assert out["dummy_variable_coloc"].sel(N_PROF=10) == 280.0
    assert out["dummy_variable_coloc"].sel(N_PROF=11) == 281.0
    assert np.isnan(out["dummy_variable_coloc"].sel(N_PROF=12))


def test_to_dataset_does_not_mutate_the_source(ds_result):
    ds_result.to_dataset()
    assert "dummy_variable_coloc" not in ds_result.points.original_data.data_vars


def test_to_dataset_without_provenance_raises(raw_result):
    with pytest.raises(ValueError):
        raw_result.to_dataset()


def test_to_dataset_refuses_a_dataframe_origin(df_result):
    with pytest.raises(TypeError, match="to_dataframe"):
        df_result.to_dataset()


# ---------- to_dataframe ----------

def test_to_dataframe_adds_coloc_column(df_result):
    out = df_result.to_dataframe()
    assert isinstance(out, pd.DataFrame)
    assert out.loc[10, "dummy_variable_coloc"] == 280.0
    assert out.loc[11, "dummy_variable_coloc"] == 281.0
    assert np.isnan(out.loc[12, "dummy_variable_coloc"])


def test_to_dataframe_does_not_mutate_the_source(df_result):
    df_result.to_dataframe()
    assert "dummy_variable_coloc" not in df_result.points.original_data.columns


def test_to_dataframe_without_provenance_raises(raw_result):
    with pytest.raises(ValueError):
        raw_result.to_dataframe()


def test_to_dataframe_refuses_a_dataset_origin(ds_result):
    with pytest.raises(TypeError, match="to_dataset"):
        ds_result.to_dataframe()
