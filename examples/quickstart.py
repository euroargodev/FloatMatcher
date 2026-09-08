"""FloatMatcher v1 — quickstart.

Colocalize a set of points (lon, lat, time) with the
daily ERA5 single-level files stored under /runtime/data/era5_daily
adapt to your local repository as the pattern. 

run with ''' uv run examples/quickstart.py '''

"""

import warnings

import numpy as np
import xarray as xr

from floatmatcher.matchup import NearestNeighbor
from floatmatcher.orchestrator import Orchestrator
from floatmatcher.products import ERA5Product
from floatmatcher.profile_loader import ProfileLoader

# silence the noise
warnings.filterwarnings("ignore")



# ---- load PointSet from arrays ----
rng = np.random.default_rng(0)
n_points = 50

lon = rng.uniform(-180.0, 180.0, n_points)
lat = rng.uniform(-60.0, 60.0, n_points)
day_offsets = rng.integers(0, 10, size=n_points).astype("timedelta64[D]")
hour_offsets = rng.integers(0, 24, size=n_points).astype("timedelta64[h]")
time = (np.datetime64("2018-01-01") + day_offsets + hour_offsets).astype("datetime64[ns]")

points = ProfileLoader.from_arrays(lon, lat, time)

print(f"\nload pointset from arrays : {len(points.lon)} points between "
      f"{points.time.min()} and {points.time.max()}")




# ---- load product with simple resolver : root path ----
# case of root path
january = ERA5Product.from_local(path=f"/runtime/data/era5_daily/2018/01")

# ::: perform matchup :::
# The orchestrator holds what to colocalize (points, variables, product).
# The method holds the constraints to be able to rerun the same orchestrator 
# with other constraints without reopening anything.
result = Orchestrator(points=points, variables=["sst"],
                              product=january).match(NearestNeighbor(max_dist_km=300,
                                                                    max_time=np.timedelta64(6, "h")))
print(f"\nload era5 product from a folder : {len(january.files_for())} files")
print(f"matchup january folder : {int(result.valid.sum())} points matched")




# ---- load product with simple resolver : case of given list of files ----
file_list = ["/runtime/data/era5_daily/2018/01/era5_single-levels_20180101.nc",
             "/runtime/data/era5_daily/2018/01/era5_single-levels_20180102.nc",
             "/runtime/data/era5_daily/2018/01/era5_single-levels_20180103.nc",
             ]
three_days = ERA5Product.from_local(path=file_list)

# ::: perform matchup :::
three_days_result = Orchestrator(points=points, variables=["sst"],
                              product=three_days).match(NearestNeighbor(max_dist_km=300,
                                                                    max_time=np.timedelta64(6, "h")))
print(f"\nload era5 product from a list : {len(three_days.files_for())} files")
print(f"matchup file_list : {int(three_days_result.valid.sum())} points matched")


# ---- load product with path + pattern ----
# this config only affects this code bloc
# if there is no pattern to find, just comment the associated code block.
ERA5_ROOT = "/runtime/data/era5_daily"
ERA5_PATTERN = "{year}/{month:02d}/era5_single-levels_{year}{month:02d}{day:02d}.nc"

# more files than days because of the time pad taking into account files before and after
product = ERA5Product.from_local(path=ERA5_ROOT, pattern=ERA5_PATTERN)

files = product.files_for(points)
print(f"\nload era5 product : {len(files)} files selected, first one: {files[0]}")
orchestrator = Orchestrator(points=points, variables=["sst"], product=product)
method = NearestNeighbor(max_dist_km=300, max_time=np.timedelta64(6, "h"))

# trigger the matchup
result = orchestrator.match(method=method)


# ---- Explore matchupresults ---- 
# returns : 
#     one array / variable 
#     distance of each point from nearest
#     time gap to the node 
#     valid mask 
#     points too far in space or in time are NaN (values) and False in valid.

n_valid = int(result.valid.sum())
print(f"\nMatchupResults : {n_valid}/{len(points.lon)} points matched within "
      f"{method.max_dist_km} km and {method.max_time_seconds / 3600:.0f} h")

kept = result.valid
print(f"   sst        : {np.round(result.values['sst'][kept][:5], 2)} ...")
print(f"   distance   : {np.round(result.distance_km[kept][:5], 1)} km")
print(f"   time gap   : {np.round(result.time_delta[kept] /3600, 1)[:5]} h")


# --- Full case with dataset origin instead of three simple arrays ---

profiles = xr.Dataset(
    {
        "LONGITUDE": (("N_PROF",), np.array([-40.0, -30.0, -20.0])),
        "LATITUDE": (("N_PROF",), np.array([35.0, 36.0, 37.0])),
        "TIME": (("N_PROF",), np.array(["2018-01-03", "2018-01-04", "2018-01-05"],
                                       dtype="datetime64[ns]")),
    },
    coords={"N_PROF": np.array([10, 11, 12])},
)

argo_points = ProfileLoader.from_xrdataset(profiles)
argo_result = Orchestrator(points=argo_points, variables=["sst"],
                           product=product).match(method=method)

enriched = argo_result.to_dataset()
print("\ndataset enriched with:", list(enriched.data_vars))
print(enriched["sst_coloc"].values)
print(enriched)
