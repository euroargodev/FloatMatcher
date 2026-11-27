# FloatMatcher

Floatmatcher is a python library for third-party dataset matchup with Argo floats 
Third-party datasets can originate from local or distant repositories.

## Documentation
Associated documentation is available running : ``` python3 -m http.server 8000 -d docs/_build/html ``` after installation 

## Installation
To install floatmatcher library, use the current git repository : 
### uv 
``` uv add "floatmatcher @ git+https://github.com/euroargodev/FloatMatcher.git@main" ```

### pip
``` pip install "git+https://github.com/euroargodev/FloatMatcher.git@main" ```

## Usage
```python
# Import the library
import floatmatcher.floatmatcher as fm
```

```python
# Instantiate builder object:
my_fm = fm.FloatMatcher() 
```

```python
# Import points:
import numpy as np

lon = np.array([-28.15, -46.51])
lat = np.array([54.48, 30.08])
time = np.array(["2013-03-18T22:47", "2014-05-18T12:22"], dtype="datetime64[ns]")

my_fm.set_points(lon, lat, time)
```


```python
# Set product 
my_fm.set_product(type="era5-hourly", 
                  source="local",
                  selected_variables=["sst"], 
                  store="/<era5_data_folder>",
                  )
```

```python 
# set method
my_fm.set_method(type="interp", radius=12, 
              time_gap=np.timedelta64(3, "h"),
              k_nearest=4, n_frames=2, kernel="idw",
              power=2
              )
```

```python
# trigger the match
my_fm.match()
```

```python
# display results 
my_fm.result
```

## Usage with argopy
An Argo dataset can be matched directly through its `.argo` accessor:

```python
import argopy
import numpy as np
import floatmatcher

ds = argopy.DataFetcher().region([-28, -25, 48, 50, 0, 50, "2018-01", "2018-06"]).data

ds_matched = ds.argo.floatmatcher.match(
    product=dict(type="era5-hourly", source="local",
                 selected_variables=["sst"], store="/<era5_data_folder>"),
    method=dict(type="nearest", radius=50, time_gap=np.timedelta64(6, "h")),
)
```

The Argo dataset comes back in the same layout as the original argopy dataset with a `sst_coloc` variable.

## Additional tutorials

Notebooks in [examples/](examples/):

1. [Quickstart](examples/01_quickstart.ipynb): the whole workflow, from points to matched values.
2. [Points](examples/02_points.ipynb): arrays, DataFrame, Dataset, Argo float with argopy.
3. [Remote CDS](examples/03_remote_cds.ipynb): CDS account, requests, cache, limits.
4. [Local archives](examples/04_local_archives.ipynb): one file, a list, a folder, a tree with a pattern, a look-up table.
5. [Methods and results](examples/05_methods_results.ipynb): nearest and interpolation, radius and time gap, several neighbors, results.
6. [Argopy accessor](examples/06_argopy_accessor.ipynb): match an argopy dataset in one call, several neighbors.

## Products, methods and options

### Products: `set_product(type, source, store, selected_variables, ...)`

`store` is where the files of the product are on disk. What to give depends on the `source`:

- When `source="local"`: the files are already on disk. `store` can then be one file, one
  directory (all `.nc` files under it and its subdirectories are opened), or a list of files
  and/or directories. With the `pattern` option, `store` must be a directory at the top of the
  tree: the pattern describes the path of the files below it to speed up file opening.
- When `source="remote-cds"`: the files are downloaded from CDS. `store` must then be one
  directory and represents the CDS cache dir. The files are downloaded there, then opened from
  there. A request already downloaded in the cache is not downloaded again.

Warning 1: the files must share the same grid and be unique: the library doesn't manage several
files having the same time: it will raise an error.

Warning 2: the time to open the files is proportional to the number of files opened. Opening a complete archive is slower than opening only the files needed (e.g. a list of files). If possible, prefer to 
build the list of files before the matchup. If not, using the `pattern` option gives similar results but only works with a daily archive tree for now.  
For example, for a float of 81 profiles and a local ERA5 archive of 3 years (one file per day): opening the entire archive (stored locally) takes about 11 s (1 096 files opened), a list of the needed files or the
directory with a `pattern` takes only about 3.5 s. This opening time might be extended in the case of shared filesystem (e.g. nfs).

```python
import cdsapi

# local archive - one file per day: every file below the directory is opened (slow)
my_fm.set_product("era5-hourly", "local", "/data/era5", ["sst"])

# local archive - one file per day: using the pattern option opens only the files concerning the times of the points (fast)
my_fm.set_product("era5-hourly", "local", "/data/era5", ["sst"],
                  pattern="{year}/{month:02d}/era5_{year}{month:02d}{day:02d}.nc")

# local archive - one file per day: an explicit list of files opens only the files given (fast)
my_fm.set_product("era5-hourly", "local",
                  ["/data/era5_20180101.nc", "/data/era5_20180102.nc"], ["sst"])

# remote archive: download files from CDS into ./cache directory (slow). Then, it  opens all the files in it (fast). In case of a re-run, if the cache has not been emptied, it opens directly the files without re-downloading it (fast)
my_fm.set_product("era5-hourly", "remote-cds", "./cache", ["sst"], client=cdsapi.Client())
```

The options allowed with each source are given in the following table:

| Option | `source="local"` | `source="remote-cds"` | Note |
|---|:---:|:---:|---|
| `store` = one file | x | | |
| `store` = one directory | x | mandatory | With `remote-cds`, a directory is required: it is used as the cache, the files are downloaded there and then opened from there. |
| `store` = list of files and/or directories | x | | |
| `selected_variables=None` (all variables) | x | | With `remote-cds`, a list of variables is required to know what to download. |
| `pattern` | optional | | `store` must then be one directory, the top of the tree of files. |
| `client` | | mandatory | Required: the client of the CDS API, `cdsapi.Client()`. |
| `clear_cache` | | optional | Optional, `False` by default. With `True`, the cache directory is emptied before downloading. |
| `time` | | optional | Optional, ERA5 only: hours of the day to download, e.g. `["00:00", "12:00"]`. All hours by default. |



The following table displays the available products with floatmatcher:

| `type` | `source` | Variables (`selected_variables`) | Description |
|---|---|---|---|
| `"era5-hourly"` | `"remote-cds"` | `sst`, `t2m`, `u10`, `v10`, `avg_slhtf`, `avg_ishf`, `avg_snswrf`, `avg_snlwrf` | ERA5 hourly reanalysis on single levels, downloaded from CDS (`reanalysis-era5-single-levels`). Grid 0.25°, hourly time axis. |
| `"era5-hourly"` | `"local"` | all the variables of the local files | ERA5 files already on disk. Grid 0.25°, hourly time axis. |
| `"lut"` | `"local"` | all the variables of the local files | Look-up table of the physiological ratio between fluorescence and chlorophyll-a, and its temporal standard deviation (`LUT_v2_202608.nc`). Grid 1°, no time axis: points without time. |



### Methods: `set_method(type, ...)`

The parameters allowed with each method are given in the following table (default value in
brackets). An empty cell means the parameter is not accepted: giving it raises an error.

| Parameter | `type="nearest"` | `type="interp"` | Note |
|---|:---:|:---:|---|
| `type` | mandatory | mandatory | `"nearest"` or `"interp"`. |
| `radius` | optional (25) | optional (25) | Maximum distance to a node of the grid, in km. Nodes beyond are invalid. |
| `time_gap` | optional (1 day) | optional (1 day) | Maximum gap to a time step, e.g. `np.timedelta64(6, "h")`. Ignored on a grid without time axis. |
| `k_nearest` | optional (1) | optional (2) | Number of nearest nodes. `nearest` keeps every neighbor. `interp` weights the neighbors to one value per point. |
| `n_frames` | optional (1) | optional (1) | Number of nearest time steps, same behavior as `k_nearest` (`t0`, `t1`...). Ignored on a grid without time axis. |
| `kernel` | | optional (`"idw"`) | Weighting function of the neighbors. Only `"idw"` (inverse distance weighting) for now. |
| `power` | | optional (2) | Power of the inverse distance in `"idw"`. |

### Outputs: `result` and augmented original objects

| Points given to `set_points` | One neighbor | Several neighbors (`nearest`) |
|---|---|---|
| arrays, `PointSet` | `result` only | `result` only, values `(n, K, N)` |
| DataFrame | column `sst_coloc` | columns `sst_coloc_s0_t0`, `sst_coloc_s0_t1`... |
| Dataset | `sst_coloc(N_PROF)` | `sst_coloc(N_PROF, k_space, n_time)`, or with `to_xarray(neigh_to_vars=True)`: `sst_coloc_s0_t0(N_PROF)`... |

`result.neigh_to_cols()` gives the same columns as the DataFrame.

`to_xarray()` adds the matched variables to a copy of the Dataset by default (`in_place=False`).
`to_xarray(in_place=True)` adds the matched variables to the original Dataset itself.

### Argopy accessor: `ds.argo.floatmatcher.match(product, method, neigh_to_vars=False, original_shape=True)`

| Argo dataset | One neighbor, `neigh_to_vars=False` | Several neighbors, `neigh_to_vars=False` | Several neighbors, `neigh_to_vars=True` |
|---|---|---|---|
| profiles (`N_PROF`) | `sst_coloc(N_PROF)` | `sst_coloc(N_PROF, k_space, n_time)` | `sst_coloc_s0_t0(N_PROF)`... |
| points (`N_POINTS`) | `sst_coloc(N_POINTS)` | error: use `neigh_to_vars=True` or `original_shape=False` | `sst_coloc_s0_t0(N_POINTS)`... |
| points (`N_POINTS`), `original_shape=False` | back in profiles, `sst_coloc(N_PROF)` | back in profiles, `sst_coloc(N_PROF, k_space, n_time)` | back in profiles, `sst_coloc_s0_t0(N_PROF)`... |

`original_shape` has no effect on a profiles dataset.
