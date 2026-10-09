import warnings

import numpy as np
import xarray as xr

import floatmatcher.floatmatcher as fm
from floatmatcher.product import ERA5Product
import mocking


# instance floatmatcher : 
my_fm = fm.FloatMatcher() 



# -------------------------------------------
# Random points case
lon, lat, time = mocking.random_positions(100)
print(type(lon))
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

# TODO : set_points_from dataframe/dataset

# 1.1 example
my_fm.set_points_from_arrays(lon, lat, time)



# -------------------------------------------
# 2) matched product 
#   2.1) product type 
#   2.2) product source (local / remote)
#   2.3) Subset variables: all or list of output product variables (add enum available)

my_fm.set_product(type="era5", 
                  source="local", # remote-zarr , remote-s3 , download-API
                  selected_variables=["sst"], 
                  path="/runtime/data/era5_daily",
                #   pattern="{year}/{month:02d}/era5_single-levels_{year}{month:02d}{day:02d}.nc"
                  )



# -------------------------------------------
# 3) matching method
    # 3.1) method selection (Nearest / interpolation)
    # 3.2) method configuration 
    #   radius (km)
    #   time
    
my_fm.set_method(type="nearest", radius=12, 
              time_gap=np.timedelta64(3, "h"),
              k_nearest=1
              )


# my_fm.download() # arriere plan (avec un state) .status

# my_fm.status() # TODO

# my_fm.match() # or resultats = my_fm.match()

# Compte rendu / visualisation etc.. 
my_fm.result

# my_fm.result_summary()
# my_fm.result_viz(header=Npoints)



print(my_fm)