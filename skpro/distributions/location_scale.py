# -*- coding: utf-8 -*-

class LocationScaleMixin():
    
    @classmethod    
    def varianceToScale(cls, variance):
         """Mapping of the distribution variance to the 'scale' parameter

         Parameters
         ----------
         variance : array-like
            Test samples

         Returns
         -------
         Mapped 'scale' parameter : array of float
         """

         raise NotImplementedError()
