from importlib.metadata import PackageNotFoundError, version

try:
    # single source of truth: the version of pyproject.toml, read from the installed package
    __version__ = version("floatmatcher")
except PackageNotFoundError:
    # local copy, not installed (same fallback as argopy)
    __version__ = "999"
