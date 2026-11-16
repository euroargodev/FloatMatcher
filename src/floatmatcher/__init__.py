from importlib.metadata import PackageNotFoundError, version

try:
    # single source of truth: the version of pyproject.toml, read from the installed package
    __version__ = version("floatmatcher")
except PackageNotFoundError:
    # local copy, not installed (same fallback as argopy)
    __version__ = "999"

# import floatmatcher accessor at floatmatcher import to have the accessor 
# directly accessible with only ```import floatmatcher```
try:
    from . import accessors  # registers ds.argo.floatmatcher if argopy is installed
except ImportError:          # argopy not installed: no accessor
    pass