# Copyright (c) 2025, YourName. All rights reserved.
# This code is licensed under the BSD-3-Clause License.

"""
Implementation of the Skew Normal distribution using skpro's ScipyAdapter.
"""

__author__ = ["Spinachboul"]

from scipy.stats import skewnorm
from skpro.distributions.adapters import ScipyAdapter


class SkewNormal(ScipyAdapter):
    r"""Skew Normal probability distribution using ScipyAdapter.

    Parameters
    ----------
    xi : float
        Location parameter of the distribution (mean shift).
    scale : float
        Scale parameter (standard deviation) of the distribution.
    shape : float
        Skewness parameter of the distribution.
    index : array-like, optional (default=None)
        Index for the distribution, for pandas-like behavior.
    columns : array-like, optional (default=None)
        Columns for the distribution, for pandas-like behavior.
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
        self._params = {"a": shape, "loc": xi, "scale": scale}
        super().__init__(
            scipy_class=skewnorm, index=index, columns=columns, **self._params
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        if parameter_set == "default":
            return {"xi": 0, "scale": 1, "shape": 5}
        return [{"xi": -1, "scale": 2, "shape": 3}, {"xi": 0, "scale": 1, "shape": -2}]