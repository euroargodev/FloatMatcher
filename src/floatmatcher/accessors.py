# accessor file register for argopy library


import xarray as xr
from argopy.extensions.utils import register_argo_accessor, ArgoAccessorExtension
from floatmatcher.floatmatcher import FloatMatcher

@register_argo_accessor('floatmatcher')
class FloatMatcherExtension(ArgoAccessorExtension):

    def match(self, product:dict|None=None, method:dict|None=None):

        # product or method are unpacked into set_product/set_method: check it is given first
        if not product or not method:
            raise AttributeError("Product or Method are not set, please fill it to use floatmatcher functionalities. ex: "
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


