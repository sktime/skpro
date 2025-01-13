# Copyright (c) 2025, YourName. All rights reserved.
# This code is licensed under the BSD-3-Clause License.

"""
Implementation of the Skew Normal distribution using skpro's BaseDistribution.
"""

__author__ = ["Spinachboul"]

import numpy as np
from scipy.stats import skewnorm
from skpro.distributions.base import BaseDistribution

class SkewNormal(BaseDistribution):
    r"""Skew Normal probability distribution.

    Parameters
    ----------
    xi : float or np.ndarray
        Location parameter of the distribution (shift of the mean).
    scale : float or np.ndarray
        Scale parameter of the distribution (standard deviation).
    shape : float or np.ndarray
        Shape (skewness) parameter of the distribution.
    index : array-like, optional (default=None)
        Index for the distribution, for pandas-like behavior.
    columns : array-like, optional (default=None)
        Columns for the distribution, for pandas-like behavior.

    Examles
    ----------
    >>> from skpro.distributions.skew_normal import SkewNormal
    >>> sn = SkewNormal(xi=0, scale=1, shape=5)
    >>> sn.mean()
    """

    _tags = {
        "authors": ["Spinachboul"],
        "maintainers": [],
        "python_version": ">=3.8",
        "python_dependencies": ["scipy"],
        "distr:measuretype": "continuous",
        "capabilities:approx": ["energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, xi, scale, shape, index=None, columns=None):
        self._xi = xi
        self._scale = scale
        self._shape = shape
        super().__init__(index=index, columns=columns)

    def _mean(self):
        delta = self._shape / np.sqrt(1 + self._shape**2)
        mean = self._xi + self._scale * delta * np.sqrt(2 / np.pi)
        return mean

    def _var(self):
        delta_sq = (self._shape**2) / (1 + self._shape**2)
        variance = self._scale**2 * (1 - (2 * delta_sq / np.pi))
        return variance

    def _pdf(self, x):
        return skewnorm.pdf(x, self._shape, loc=self._xi, scale=self._scale)

    def _log_pdf(self, x):
        return skewnorm.logpdf(x, self._shape, loc=self._xi, scale=self._scale)

    def _cdf(self, x):
        return skewnorm.cdf(x, self._shape, loc=self._xi, scale=self._scale)

    def _ppf(self, p):
        return skewnorm.ppf(p, self._shape, loc=self._xi, scale=self._scale)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        if parameter_set == "default":
            return {"xi": 0, "scale": 1, "shape": 5}
        return [{"xi": -1, "scale": 2, "shape": 3}, {"xi": 0, "scale": 1, "shape": -2}]