# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Skew-Normal probability distribution."""

__author__ = ["Spinachboul"]

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

from skpro.distributions.adapters.scipy._distribution import _ScipyAdapter

class SkewNormal(_ScipyAdapter):
    r"""Skew-normal distribution (skpro native).

    This distribution has a shape (skewness) parameter `shape`, location `loc`,
    and scale `scale`. It generalizes the normal distribution for skewed data.

    Parameters
    ----------
    shape : float, default=0
        Skewness parameter, where 0 gives a normal distribution.
    loc : float, default=0
        Location (mean) parameter.
    scale : float, default=1
        Scale (standard deviation) parameter, must be positive.
    index : pd.Index, optional
        Index of the distribution.
    columns : pd.Index, optional
        Columns of the distribution.

    Examples
    --------
    >>> from skpro.distributions.skew_normal import SkewNormal
    >>> sn = SkewNormal(shape=4, loc=0, scale=1)
    >>> sn.mean()
    """

    _tags = {
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, shape=0, loc=0, scale=1, index=None, columns=None):
        self._shape = shape
        self._loc = loc
        self._scale = scale
        super().__init__(index=index, columns=columns)

    @property
    def shape(self):
        """Skewness parameter."""
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def loc(self):
        """Location parameter."""
        return self._loc

    @loc.setter
    def loc(self, value):
        self._loc = value

    @property
    def scale(self):
        """Scale parameter."""
        return self._scale

    @scale.setter
    def scale(self, value):
        if value <= 0:
            raise ValueError("Scale must be positive.")
        self._scale = value

    def _get_scipy_object(self):
        """Return the scipy.stats.skewnorm object."""
        return skewnorm

    def _get_scipy_param(self):
        """Return parameters for the skew-normal distribution."""
        return self.shape, {"loc": self.loc, "scale": self.scale}

    def _mean(self):
        """Return the mean of the skew-normal distribution."""
        shape, loc, scale = self.shape, self.loc, self.scale
        delta = shape / np.sqrt(1 + shape**2)
        return loc + scale * delta * np.sqrt(2 / np.pi)

    def _var(self):
        """Return the variance of the skew-normal distribution."""
        shape, scale = self.shape, self.scale
        delta_squared = shape**2 / (1 + shape**2)
        return scale**2 * (1 - 2 * delta_squared / np.pi)

    def _pdf(self, x):
        """Return the PDF of the skew-normal distribution."""
        return skewnorm.pdf(x, self.shape, loc=self.loc, scale=self.scale)

    def _log_pdf(self, x):
        """Return the log-PDF of the skew-normal distribution."""
        return skewnorm.logpdf(x, self.shape, loc=self.loc, scale=self.scale)

    def _cdf(self, x):
        """Return the CDF of the skew-normal distribution."""
        return skewnorm.cdf(x, self.shape, loc=self.loc, scale=self.scale)

    def _ppf(self, p):
        """Return the PPF of the skew-normal distribution."""
        return skewnorm.ppf(p, self.shape, loc=self.loc, scale=self.scale)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"shape": 4, "loc": 0, "scale": 1}
        params2 = {"shape": -3, "loc": 5, "scale": 2, "index": pd.Index([0, 1, 2])}
        return [params1, params2]