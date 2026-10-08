

# instance floatmatcher : 
import floatmatcher as fm
my_fm = fm.new() 

# enter input parameters : 
# 1) match positions : lon lat time
#   1.1) custom series (3 arrays)
#   1.2) dataframe + mapping dictionnary
#   1.3) dataset xr + mapping dictionnary
#   1.4) argopy (no mapping needed) K
my_serie=...
my_fm.input_positions.from_arrays(my_serie)

# 2) matched product 
#   2.1) product type 
#   2.2) product source (local / remote)
#   2.3) Subset variables: all or list of output product variables (add enum available)

# 3) matching method
    # 3.1) method selection (Nearest / interpolation)
    # 3.2) method configuration 
    #   radius (km)
    #   time
    #