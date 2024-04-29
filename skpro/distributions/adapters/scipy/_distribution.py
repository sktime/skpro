from typing import Union

from skpro.distributions.base import BaseDistribution

import pandas as pd
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats import rv_continuous

__all__ = ["_ScipyAdapter"]

class _ScipyAdapter(BaseDistribution):
    _distribution_attr = "_dist"

    def __init__(self, index=None, columns=None):
        obj = self._get_scipy_object()
        setattr(self, self._distribution_attr, obj)
        super().__init__(index, columns)

    def _get_scipy_object(self) -> Union[rv_continuous, rv_discrete]:
        raise NotImplementedError("abstract method")

    def _get_scipy_param(self) -> dict:
        raise NotImplementedError("abstract method")

    def _mean(self, *args, **kwds):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.mean(*args, **kwds, **params)

    def _var(self, *args, **kwds):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.var(*args, **kwds, **params)

    def _pdf(self, x: pd.DataFrame, *args, **kwds):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.pdf(x.values, *args, **kwds, **params)

    def _log_pdf(self, x: pd.DataFrame, *args, **kwds):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.logpdf(x.values, *args, **kwds, **params)

    def _cdf(self, x: pd.DataFrame, *args, **kwds):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.cdf(x.values, *args, **kwds, **params)

    def _ppf(self, q: pd.DataFrame, *args, **kwds):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        params = self._get_scipy_param()
        return obj.ppf(q.values, *args, **kwds, **params)
