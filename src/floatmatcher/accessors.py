# accessor file register for argopy library

from argopy.extensions.utils import register_argo_accessor, ArgoAccessorExtension
from floatmatcher.floatmatcher import FloatMatcher

@register_argo_accessor('floatmatcher')
class FloatMatcherExtension(ArgoAccessorExtension):
    """Match an Argo dataset with a gridded product, with FloatMatcher

    The positions and times of the profiles are matched with a gridded product (ERA5, LUT),
    from local files or from CDS. The matched values are added to the dataset as
    ``<variable>_coloc`` variables.

    See Also
    --------
    :meth:`~xarray.Dataset.argo.floatmatcher.match`, :class:`.FloatMatcher`

    Examples
    --------
    Load Argo data:

    .. code-block:: python

        import argopy
        import numpy as np
        import floatmatcher

        ds = argopy.DataFetcher().region([-28, -25, 48, 50, 0, 50, '2018-01', '2018-06']).data

    Match them with a product, here the sea surface temperature of local ERA5 files, and a method:

    .. code-block:: python

        ds.argo.floatmatcher.match(
            product=dict(type='era5-hourly', source='local',
                         selected_variables=['sst'], path='/<era5_data_folder>'),
            method=dict(type='nearest', radius=50, time_gap=np.timedelta64(6, 'h')),
        )

    The dataset can be in points (``N_POINTS``) or profiles (``N_PROF``) layout: it is matched
    by profile and comes back in the same layout.
    """

    def match(self, product:dict|None=None, method:dict|None=None):
        """Match the profiles or points of the dataset with a gridded product

        Parameters
        ----------
        product: dict
            Parameters of :meth:`.FloatMatcher.set_product`: ``type``, ``source``, ``path``,
            ``selected_variables`` and the parameters of the source.
        method: dict
            Parameters of :meth:`.FloatMatcher.set_method`: ``type`` and the parameters of the
            method.

        Returns
        -------
        :class:`xarray.Dataset`
            The dataset with a ``<variable>_coloc`` variable per matched variable, in the same
            layout as the input (``N_POINTS`` or ``N_PROF``).

        Raises
        ------
        AttributeError
            If ``product`` or ``method`` is not given.
        ValueError
            If the product, the source or the method is unknown.
        TypeError
            If a parameter of the source or of the method is unknown, or a required one is
            missing.
        """

        # product or method are unpacked into set_product/set_method: check it is given first
        if not product or not method:
            raise AttributeError("Product or Method are not set, "
                "please fill it to use floatmatcher functionalities. ex: "
                """f.data.argo.floatmatcher.match(
                product=dict(type='era5-hourly',
                            source='local',
                            selected_variables=['sst'],
                            path=root
                            ),
                method=dict(type='nearest',
                            radius=50,
                            time_gap=np.timedelta64(6, 'h')
                            ),
            )
            """)

        # init floatmatcher object
        fm = FloatMatcher()

        # set points
        marker =  "N_POINTS" in self._obj.dims

        if marker:
            dataset = self._obj.argo.point2profile()
        else:
            dataset = self._obj

        fm.set_points(dataset)

        # set product
        fm.set_product(**product)

        # set method
        fm.set_method(**method)

        # match
        fm.match()

        # reinject
        ds_out = fm.reinject()

        if marker:
            ds_out = ds_out.argo.profile2point()

        return ds_out


