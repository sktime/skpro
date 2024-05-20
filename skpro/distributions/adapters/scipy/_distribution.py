# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Adapter for Scipy Distributions."""

__author__ = ["malikrafsan"]

from typing import Union

import pandas as pd
from scipy.stats import rv_continuous, rv_discrete

from skpro.distributions.base import BaseDistribution

__all__ = ["_ScipyAdapter"]


class _ScipyAdapter(BaseDistribution):
    """Adapter for scipy distributions.

    This class is an adapter for scipy distributions. It provides a common
    interface for all scipy distributions. The class is abstract
    and should not be instantiated directly.
    """

    _distribution_attr = "_dist"
    _tags = {
        "object_type": "distribution",
        "distr:paramtype": "parametric",
    }

    def __init__(self, index=None, columns=None):
        obj = self._get_scipy_object()
        setattr(self, self._distribution_attr, obj)
        super().__init__(index, columns)

    def _get_scipy_object(self) -> Union[rv_continuous, rv_discrete]:
        """Abstract method to get the scipy distribution object.

        Should import the scipy distribution object and return it.
        """
        raise NotImplementedError("abstract method")

    def _get_scipy_param(self):
        """Abstract method to get the scipy distribution parameters.

        Should return a tuple with two elements: a list of positional arguments (args)
        and a dictionary of keyword arguments (kwds).
        """
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

    def _ppf(self, p: pd.DataFrame):
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        args, kwds = self._get_scipy_param()
        return obj.ppf(p, *args, **kwds)

    def _pmf(self, x: pd.DataFrame):
        """Return the probability mass function evaluated at x."""
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        args, kwds = self._get_scipy_param()
        return obj.pmf(x, *args, **kwds)

    def _log_pmf(self, x: pd.DataFrame):
        """Return the log of the probability mass function evaluated at x."""
        obj: Union[rv_continuous, rv_discrete] = getattr(self, self._distribution_attr)
        args, kwds = self._get_scipy_param()
        return obj.logpmf(x, *args, **kwds)
