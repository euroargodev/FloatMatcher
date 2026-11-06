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
                  client=cdsapi.Client(),     
                  clear_cache=True,
                  time=["00:00", "12:00"]
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





## Additional tutorials



