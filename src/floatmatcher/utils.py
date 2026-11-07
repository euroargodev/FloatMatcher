# utils.py

# All general methods are grouped here to avoid "@staticmethod" 
# in classes and lightning the writings. Furthermore, reusable 
# in other files without whole class import

import xarray as xr
from .exceptions import ProfileFormatError
from numpy.typing import NDArray
import inspect
from collections.abc import Iterable
from typing import Any


def _select_variables(ds: xr.Dataset, variables: str | list[str] | None = None) -> xr.Dataset:
    """Keep only the requested variables

    Parameters
    ----------
    ds: xarray.Dataset
    variables: str, list of str or None
        Variables to keep. None keeps all.

    Returns
    -------
    :class:`xarray.Dataset`
        The data are not copied.

    Raises
    ------
    ValueError
        If a variable is not in the dataset.
    """
    if variables is None:
        return ds
    
    wanted = [variables] if isinstance(variables, str) else list(variables)
    missing = [v for v in wanted if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"variables not found in the dataset: {missing}. "
            f"Available: {list(ds.data_vars)}."
        )
    return ds[wanted]


def _find_key(ds: xr.Dataset, name: str) -> str:
    """Name of a coordinate or variable, checked to exist in the dataset

    Parameters
    ----------
    ds: xarray.Dataset
    name: str

    Returns
    -------
    str
        ``name`` itself.

    Raises
    ------
    ProfileFormatError
        If no coordinate or variable has this name.
    """
    if name in ds.coords or name in ds.variables:
        return name
    raise ProfileFormatError(f"No dataset name matches: {name}")


def _get(ds: xr.Dataset, name: str) -> xr.DataArray:
    """DataArray of a coordinate or variable

    Parameters
    ----------
    ds: xarray.Dataset
    name: str

    Returns
    -------
    :class:`xarray.DataArray`

    Raises
    ------
    ProfileFormatError
        If no coordinate or variable has this name.
    """
    return ds[_find_key(ds, name)]


def _extract(ds: xr.Dataset, name: str) -> NDArray[Any]:
    """Values of a coordinate or variable, as a NumPy array

    Parameters
    ----------
    ds: xarray.Dataset
    name: str

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ProfileFormatError
        If no coordinate or variable has this name.
    """
    return _get(ds, name).values


def _signature_defaults(cls: type, names: Iterable[str]) -> dict[str, Any]:
    """Default values of parameters of a class, read in the signature of its ``__init__``

    Parameters
    ----------
    cls: type
    names: iterable of str
        Names of parameters of ``cls.__init__`` that have a default value.

    Returns
    -------
    dict
        Name -> default value.
    """
    parameters = inspect.signature(cls).parameters
    return {name: parameters[name].default for name in sorted(names)}


def _check_params(kind: str, name: str, given: Iterable[str],
                  required: set[str], optional: dict[str, Any]) -> None:
    """Check the parameters given to a source or a method, before creating it

    Parameters
    ----------
    kind: str
        Kind of object, ``"source"`` or ``"method"``, for the messages.
    name: str
        Name of the object, for the messages.
    given: iterable of str
        Names of the parameters given.
    required: set of str
        Parameters that must be given.
    optional: dict
        Parameters that have a default value -> that value.

    Raises
    ------
    TypeError
        If a required parameter is missing, or if a parameter is unknown.
    """
    given_names = set(given)
    known_names = required | set(optional)

    missing_names = sorted(required - given_names)
    unknown_names = sorted(given_names - known_names)
    if not missing_names and not unknown_names:
        return

    # what the object expects, written in the message
    expected = (f"required: {sorted(required)}, "
                f"optional params: {optional}, "
                f"consider filling at least the required params")

    if missing_names:
        raise TypeError(f"{kind} {name!r} needs {missing_names} ({expected})")
    raise TypeError(f"{kind} {name!r} does not take {unknown_names} ({expected})")


def _indented_repr(obj: object, indent: int = 4) -> str:
    """Indented listing of the attributes of an object

    Attributes that are floatmatcher objects are listed one level deeper. Used by the
    ``__repr__`` of the products, sources and methods.

    Parameters
    ----------
    obj: object
    indent: int, default=4
        Number of spaces of the first level.

    Returns
    -------
    str
    """
    pad = " " * indent
    text = f"\n{pad}type: {type(obj).__name__}"
    for name, value in obj.__dict__.items():
        if type(value).__module__.startswith("floatmatcher"):
            text += f"\n{pad}{name}:{_indented_repr(value, indent + 4)}"
        else:
            text += f"\n{pad}{name}: {value}"
    return text
