# FloatMatcher

Floatmatcher is a a python library for third-party dataset matchup with Argo floats 
Third-party datasets can originate from local or distant repositories.

## Documentation
Asociated documentation is available running : ``` python3 -m http.server 8000 -d docs/_build/html ``` after installation 

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
# Instanciate builder object:
my_fm = fm.FloatMatcher() 
```

```python
# Import points:
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
                  path="/<era5_data_folder>",
                  )
```

```python 
# set method
import numpy as np
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
                 selected_variables=["sst"], path="/<era5_data_folder>"),
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

### Products: `set_product(type, source, path, selected_variables, ...)`

| `type` | `source` | Source parameters | Grid |
|---|---|---|---|
| `"era5-hourly"` | `"local"` | `pattern` (optional) | 0.25°, hourly time axis |
| `"era5-hourly"` | `"remote-cds"` | `client` (required), `clear_cache` (False), `time` (all hours) | 0.25°, hourly time axis |
| `"lut"` | `"local"` | `pattern` (optional) | 1°, no time axis: points without time |

| `type` | `source` | Variable (`selected_variables`) | Description |
|---|---|---|---|
| `"era5-hourly"` | `"remote-cds"` | `sst` | sea surface temperature |
| | | `t2m` | 2 m temperature |
| | | `u10`, `v10` | 10 m u and v components of wind |
| | | `avg_slhtf` | mean surface latent heat flux |
| | | `avg_ishf` | mean surface sensible heat flux |
| | | `avg_snswrf` | mean surface net short-wave radiation flux |
| | | `avg_snlwrf` | mean surface net long-wave radiation flux |
| `"era5-hourly"` | `"local"` | the variables of the local files | |
| `"lut"` | `"local"` | `fluorescence_chlorophyll_ratio` | physiological ratio between fluorescence and chlorophyll-a (`LUT_v2_202608.nc`) |
| | | `sd_fluorescence_chlorophyll_ratio` | its temporal standard deviation |

### Methods: `set_method(type, ...)`

| Parameter | `"nearest"` | `"interp"` |
|---|---|---|
| `radius` (km) | 25 | 25 |
| `time_gap` | 1 day | 1 day |
| `k_nearest`: nearest nodes | 1 | 2 |
| `n_frames`: nearest time steps | 1 | 1 |
| `kernel`, `power` | - | `"idw"`, 2 |

`time_gap` and `n_frames` are ignored on a grid without time axis. `interp` weights its neighbors into one value per point; `nearest` with `k_nearest` or `n_frames` above 1 keeps every neighbor, sorted from the nearest (`s0`, `t0`).

### Outputs: `result` and augmented original objects

| Points given to `set_points` | One neighbor | Several neighbors (`nearest`) |
|---|---|---|
| arrays, `PointSet` | `result` only | `result` only, values `(n, K, N)` |
| DataFrame | column `sst_coloc` | columns `sst_coloc_s0_t0`, `sst_coloc_s0_t1`... |
| Dataset | `sst_coloc(N_PROF)` | `sst_coloc(N_PROF, k_space, n_time)`, or with `to_xarray(neigh_to_vars=True)`: `sst_coloc_s0_t0(N_PROF)`... |

`result.neigh_to_cols()` gives the same columns as the DataFrame.

### Argopy accessor: `ds.argo.floatmatcher.match(product, method, neigh_to_vars=False, original_shape=True)`

| Argo dataset | One neighbor, `neigh_to_vars=False` | Several neighbors, `neigh_to_vars=False` | Several neighbors, `neigh_to_vars=True` |
|---|---|---|---|
| profiles (`N_PROF`) | `sst_coloc(N_PROF)` | `sst_coloc(N_PROF, k_space, n_time)` | `sst_coloc_s0_t0(N_PROF)`... |
| points (`N_POINTS`) | `sst_coloc(N_POINTS)` | error: use `neigh_to_vars=True` or `original_shape=False` | `sst_coloc_s0_t0(N_POINTS)`... |
| points (`N_POINTS`), `original_shape=False` | back in profiles, `sst_coloc(N_PROF)` | back in profiles, `sst_coloc(N_PROF, k_space, n_time)` | back in profiles, `sst_coloc_s0_t0(N_PROF)`... |

`original_shape` has no effect on a profiles dataset.
