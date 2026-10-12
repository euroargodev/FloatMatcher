import tomllib
from pathlib import Path

from floatmatcher import __version__


def test_version_comes_from_pyproject() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert __version__ == pyproject["project"]["version"]
