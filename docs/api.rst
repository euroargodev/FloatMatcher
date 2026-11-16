API reference
=============

This page is generated from the docstrings.

Matchup
-------

.. currentmodule:: floatmatcher.floatmatcher

.. autosummary::
   :toctree: generated/

   FloatMatcher

Argopy accessor
---------------

With argopy object, an Argo dataset is matched directly through its ``.argo`` accessor.

.. currentmodule:: xarray

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Dataset.argo.floatmatcher
   Dataset.argo.floatmatcher.match

Points and results
------------------

.. currentmodule:: floatmatcher.pointset

.. autosummary::
   :toctree: generated/

   PointSet

.. currentmodule:: floatmatcher.matchup_results

.. autosummary::
   :toctree: generated/

   MatchupResult

Errors
------

.. currentmodule:: floatmatcher.exceptions

.. autosummary::
   :toctree: generated/

   FloatMatcherError
   ProfileFormatError

Extending FloatMatcher
----------------------

Products, sources and methods are classes registered by name, so that ``set_product`` and
``set_method`` can find them.

.. currentmodule:: floatmatcher.product

.. autosummary::
   :toctree: generated/

   Product
   ERA5Product
   LUTProduct
   to_standard
   available_products

.. currentmodule:: floatmatcher.sources

.. autosummary::
   :toctree: generated/

   Source
   LocalSource
   CDSSource
   available_sources

.. currentmodule:: floatmatcher.methods

.. autosummary::
   :toctree: generated/

   Method
   NearestNeighbor
   Interpolator
   available_methods
