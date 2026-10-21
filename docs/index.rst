FloatMatcher
============

FloatMatcher matches positions, such as Argo profiles, with a gridded product such as the
ERA5 reanalysis: each position gets the values of the nearest node of the grid, within a
distance and a time gap.

.. code-block:: python

    import numpy as np
    from floatmatcher.floatmatcher import FloatMatcher

    fm = FloatMatcher()
    fm.set_points(lon, lat, time)                      # or a DataFrame, or an xarray Dataset
    fm.set_product("era5", "local", "/data/era5", ["sst"],
                   pattern="{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc")
    fm.set_method("nearest", radius=50, time_gap=np.timedelta64(3, "h"))
    fm.match()
    fm.result          # matched values, distances and time gaps
    fm.reinject()      # the input DataFrame or Dataset, with <name>_coloc added

.. toctree::
   :maxdepth: 2

   api
