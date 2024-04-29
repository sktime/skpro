from skpro.distributions.base import BaseDistribution

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_generic
from scipy.stats import rv_continuous

__all__ = ["_ScipyAdapter", "_ScipyContinuousAdapter"]

class _ScipyAdapter(BaseDistribution):
    _distribution_attr = "_dist"

    def __init__(self, index=None, columns=None):
        obj = self._get_scipy_object()
        setattr(self, self._distribution_attr, obj)
        super().__init__(index, columns)

    def _get_scipy_object(self) -> rv_generic:
        raise NotImplementedError("abstract method")

    def _get_scipy_param(self) -> dict:
        raise NotImplementedError("abstract method")

    def _mean(self, *args, **kwds):
        obj: rv_generic = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.mean(*args, **kwds, **params)

    def _var(self, *args, **kwds):
        obj: rv_generic = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.var(*args, **kwds, **params)


class _ScipyContinuousAdapter(_ScipyAdapter):
    def __init__(self, index=None, columns=None):
        super().__init__(index, columns)
    
    def _get_scipy_object(self) -> rv_continuous:
        raise NotImplementedError("abstract method")

    def _pdf(self, x: pd.DataFrame, *args, **kwds):
        obj: rv_continuous = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.pdf(x.values, *args, **kwds, **params)

    def _log_pdf(self, x: pd.DataFrame, *args, **kwds):
        obj: rv_continuous = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.logpdf(x.values, *args, **kwds, **params)

    def _cdf(self, x: pd.DataFrame, *args, **kwds):
        obj: rv_continuous = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.cdf(x.values, *args, **kwds, **params)

    def _ppf(self, q: pd.DataFrame, *args, **kwds):
        obj: rv_continuous = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.ppf(q.values, *args, **kwds, **params)
