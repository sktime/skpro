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

    def _get_scipy_param(self) -> tuple[list, dict]:
        raise NotImplementedError("abstract method")

    def _mean(self):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        args, kwds = self._get_scipy_param()
        return obj.mean(*args, **kwds)

    def _var(self):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        args, kwds = self._get_scipy_param()
        return obj.var(*args, **kwds)

    def _pdf(self, x: pd.DataFrame):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        args, kwds = self._get_scipy_param()
        return obj.pdf(x, *args, **kwds)

    def _log_pdf(self, x: pd.DataFrame):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        args, kwds = self._get_scipy_param()
        return obj.logpdf(x, *args, **kwds)

    def _cdf(self, x: pd.DataFrame):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        args, kwds = self._get_scipy_param()
        return obj.cdf(x, *args, **kwds)

    def _ppf(self, q: pd.DataFrame):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        args, kwds = self._get_scipy_param()
        return obj.ppf(q, *args, **kwds)
