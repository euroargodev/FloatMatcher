
# Imports
import xarray as xr
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Imports of lib components
from floatmatcher.orchestrator import Orchestrator
from floatmatcher.products import ERA5Product
from floatmatcher.profile_loader import ProfileLoader
from floatmatcher.nearest import NearestNeighbor


file_path = "/runtime/data/era5/data_stream-oper_stepType-instant.nc"


# Explore dataset
ds = xr.open_dataset(file_path)
# print("time:", ds.valid_time.values[0], "->", ds.valid_time.values[-1])
# print("time:", ds.valid_time.values)
# print("lon :", float(ds.longitude.min()), "->", float(ds.longitude.max()))
# print("lat :", float(ds.latitude.values[0]), "->", float(ds.latitude.values[-1]))
# print("vars:", list(ds.data_vars))



# SetUp Points
points = ProfileLoader.from_arrays(
    lon=[-45.13, 10.37, 100.62],      # décalés des nœuds 0.25°
    lat=[30.11, 40.42, -20.08],
    time=np.array(["2008-01-09T03", "2017-03-17T08", "2023-02-01T14"],
                  dtype="datetime64[ns]"),   # heures décalées aussi
)

# print("\n--------------------")
# orch = Orchestrator(ERA5Product(), Interpolation("linear"))
# res = orch.colocalize(file_path, points)
# print("valid :", res.valid)
# print("values:", res.values)


print("\n--------------------")
# orch_n = Orchestrator(ERA5Product(), NearestNeighbor())
# res_n = orch_n.colocalize(file_path, points)
# print("nearest valid :", res_n.valid)
# print("nearest values:", res_n.values)


orch = Orchestrator(ERA5Product(), NearestNeighbor(max_files=1))
res = orch.colocalize(file_path, points, variables="sst")          # une seule
res = orch.colocalize(file_path, points, variables=["sst", "t2m"]) # plusieurs
res = orch.colocalize(file_path, points)                            # toutes (défaut)

orch.colocalize(file_path, points, variables="xxx")   # -> ValueError listant u10,v10,t2m,sst