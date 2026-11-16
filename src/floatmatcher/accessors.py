# accessor file register for argopy library

import argopy
import numpy as np 
from argopy.extensions.utils import register_argo_accessor, ArgoAccessorExtension

@register_argo_accessor('floatmatcher')
class FloatMatcherExtension(ArgoAccessorExtension):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._uid = argopy.utils.to_list(np.unique(self._obj["PLATFORM_NUMBER"].values))

    @property
    def wmo(self):
        return self._uid

    def match(self, product=None, method=None):
        print("coucou")
