# Configuration file for the Sphinx documentation builder.
# Build: uv run --group docs sphinx-build -W -b html docs docs/_build/html

import floatmatcher

project = "FloatMatcher"
author = "Clement Weber, Guillaume Maze, Kevin Ballem, Alban Sizun, Virginie Racapé"
copyright = "EUPL v1.2"
release = floatmatcher.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# API pages are generated from the docstrings (numpydoc style, like argopy)
autosummary_generate = True
numpydoc_show_class_members = False

# readable names: "FloatMatcher" rather than "floatmatcher.floatmatcher.FloatMatcher"
add_module_names = False                  # signatures
toc_object_entries_show_parents = "hide"  # "On this page": set_points, not FloatMatcher.set_points
autodoc_member_order = "bysource"         # methods in the order of the code (the user workflow)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}

html_theme = "sphinx_book_theme"
html_title = "FloatMatcher"
html_theme_options = {
    "repository_url": "https://github.com/euroargodev/FloatMatcher",
    "use_repository_button": True,
}
