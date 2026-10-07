# tests/test_results.py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from floatmatcher.matchup_results import MatchupResult
from floatmatcher.pointset import PointSet


@pytest.fixture
def result():
    """ MatchupResult object - 3 profiles """
    return MatchupResult(
        values={"dummy_variable": np.array([280.0, 281.0, np.nan])},
        distance_km=np.array([1.0, 2.0, np.nan]),
        time_delta=np.array([0.1, 0.2, np.nan]),
        valid=np.array([True, True, False]),
    )

@pytest.fixture
def ds_points():
    """ PointSet - 3 profiles from a Dataset """
    ds = xr.Dataset(
        {"LONGITUDE": (("N_PROF",), np.array([-40.0, -30.0, -20.0])),
         "LATITUDE": (("N_PROF",), np.array([35.0, 36.0, 37.0])),
         "TIME": (("N_PROF",), np.array(["2018-01-03", "2018-01-01", "2018-01-02"],
                                        dtype="datetime64[ns]"))},
        coords={"N_PROF": np.array([10, 11, 12])},
    )
    return PointSet(
        lon=ds["LONGITUDE"].values,
        lat=ds["LATITUDE"].values,
        time=ds["TIME"].values,
        points_dim="N_PROF",
        original_data=ds,
    )

@pytest.fixture
def df_points():
    """ PointSet - 3 profiles from a DataFrame """
    df = pd.DataFrame(
        {"longitude": [-40.0, -30.0, -20.0],
         "latitude": [35.0, 36.0, 37.0],
         "date": pd.to_datetime(["2018-01-03", "2018-01-01", "2018-01-02"])},
        index=pd.Index([10, 11, 12], name="profile"),
    )
    return PointSet.from_dataframe(df)

@pytest.fixture
def raw_points():
    """Raw-array points carry no origin, so they cannot be reinjected."""
    return PointSet(np.array([-45.0, -44.0, -43.0]), np.array([32.0, 33.0, 34.0]),
                    np.array(["2015-01-01", "2015-01-02", "2015-01-03"],
                             dtype="datetime64[ns]"))

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

def test_to_dataset_adds_coloc_variable(result, ds_points):
    out = result._to_dataset(ds_points)
    assert out["dummy_variable_coloc"].dims == ("N_PROF",)
    assert out["dummy_variable_coloc"].sel(N_PROF=10) == 280.0
    assert out["dummy_variable_coloc"].sel(N_PROF=11) == 281.0
    assert np.isnan(out["dummy_variable_coloc"].sel(N_PROF=12))


def test_to_dataset_does_not_mutate_the_source(result, ds_points):
    result._to_dataset(ds_points)
    assert "dummy_variable_coloc" not in ds_points.original_data.data_vars


def test_to_dataset_without_provenance_raises(result, raw_points):
    with pytest.raises(ValueError):
        result._to_dataset(raw_points)


def test_to_dataset_refuses_a_dataframe_origin(result, df_points):
    with pytest.raises(TypeError, match="to_dataframe"):
        result._to_dataset(df_points)


# ---------- to_dataframe ----------

def test_to_dataframe_adds_coloc_column(result, df_points):
    out = result._to_dataframe(df_points)
    assert isinstance(out, pd.DataFrame)
    assert out.loc[10, "dummy_variable_coloc"] == 280.0
    assert out.loc[11, "dummy_variable_coloc"] == 281.0
    assert np.isnan(out.loc[12, "dummy_variable_coloc"])


def test_to_dataframe_does_not_mutate_the_source(result, df_points):
    result._to_dataframe(df_points)
    assert "dummy_variable_coloc" not in df_points.original_data.columns


def test_to_dataframe_without_provenance_raises(result, raw_points):
    with pytest.raises(ValueError):
        result._to_dataframe(raw_points)


def test_to_dataframe_refuses_a_dataset_origin(result, ds_points):
    with pytest.raises(TypeError, match="to_dataset"):
        result._to_dataframe(ds_points)
