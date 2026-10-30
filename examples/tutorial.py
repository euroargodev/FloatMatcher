import warnings

import numpy as np
import xarray as xr

import floatmatcher.floatmatcher as fm
from floatmatcher.product import ERA5Product
from floatmatcher.mocking import FakeCDSClient
import cdsapi


# instance floatmatcher : 
my_fm = fm.FloatMatcher() 



# -------------------------------------------
# Random points case
# lon, lat, time = mocking.random_positions(100)
# print(type(lon))

# 10 points in the North Atlantic, between 2002 and 2004
lon = np.array([-28.15, -46.51, -57.95, -59.17, -19.34, -14.36, -29.67, -23.53, -32.82, -13.25])
lat = np.array([54.48, 30.08, 55.72, 31.01, 51.89, 35.27, 55.90, 46.24, 38.99, 42.68])
time = np.array([
    "2003-03-18T22:47", "2002-02-01T00:55", "2002-01-06T20:47", "2002-05-17T05:08",
    "2002-01-10T01:54", "2004-01-06T00:06", "2003-07-31T01:50", "2003-12-11T07:40",
    "2002-10-10T00:00", "2003-11-06T11:05",
], dtype="datetime64[ns]")


# positions = np.array([lon, lat, time])
# points_from_arrays = ProfileLoader.from_arrays(lon, lat, time)

# # Xarray dataset case
# ds_profiles = mocking.random_dataset()
# points_from_dataset = ProfileLoader.from_xrdataset(ds_profiles)



# -------------------------------------------
# enter input parameters : 
# 1) match positions : lon lat time
#   1.1) custom series (3 arrays) 
#   1.2) dataframe + mapping dictionnary
#   1.3) dataset xr + mapping dictionnary
# #   1.4) argopy (no mapping needed) K


# 1.1 example
my_fm.set_points(lon, lat, time)



# -------------------------------------------
# 2) matched product 
#   2.1) product type 
#   2.2) product source (local / remote)
#   2.3) Subset variables: all or list of output product variables (add enum available)

# my_fm.set_product(type="era5", 
#                   source="local", # remote-zarr , remote-s3 , download-API
#                   selected_variables=["sst"], 
#                   path="/runtime/data/era5_daily",
#                 #   pattern="{year}/{month:02d}/era5_single-levels_{year}{month:02d}{day:02d}.nc"
#                   )


my_fm.set_product(type="era5", 
                  source="remote-cds", # remote-zarr , remote-s3 , download-API
                  selected_variables=["sst"], 
                  path="/runtime/data/cache",
                  client=FakeCDSClient(limit=1000),   # no network; toy cost, see mocking.py
                #   client=cdsapi.Client(),              # real CDS, key in ~/.cdsapirc
                  )


my_fm.build_request()
print(my_fm)
# my_fm.download()

exit()
# -------------------------------------------
# 3) matching method
    # 3.1) method selection (Nearest / interpolation)
    # 3.2) method configuration 
    #   radius (km)
    #   time
    
my_fm.set_method(type="interp", radius=12, 
              time_gap=np.timedelta64(3, "h"),
              k_nearest=4, n_frames=2, kernel="idw",
              power=2
              )


# my_fm.download() # arriere plan (avec un state) .status

# my_fm.status() # TODO

my_fm.match()

# Compte rendu / visualisation etc.. 
my_fm.result

# my_fm.result_summary()
# my_fm.result_viz(header=Npoints)



print(my_fm)