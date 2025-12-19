# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Exponential probability distribution."""

__author__ = ["ShreeshaM07"]

import numpy as np
import pandas as pd
from scipy.stats import expon, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Exponential(_ScipyAdapter):
    r"""Exponential Distribution.

    Most methods wrap ``scipy.stats.expon``.

    The Exponential distribution is parametrized by mean :math:`\mu` and
    scale :math:`b`, such that the pdf is

    .. math:: f(x) = \lambda*\exp\left(-\lambda*x\right)

    The rate :math:`\lambda` is represented by the parameter ``rate``,

    Parameter
    ---------
    rate : float or array of float (1D or 2D)
        rate of the distribution
        rate = 1/scale
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.exponential import Exponential
    >>> d = Exponential(rate=2)

    Energy computations (exact, closed-form formulas):

    >>> d.energy()  # self-energy: E|X-Y| = 2/lambda
    """

    _tags = {
        "capabilities:approx": ["ppf", "pdfnorm"],
        "capabilities:exact": [
            "mean",
            "var",
            "pdf",
            "log_pdf",
            "cdf",
            "energy",
        ],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
    }

    def __init__(self, rate, index=None, columns=None):
        self.rate = rate

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return expon

    def _get_scipy_param(self):
        rate = self._bc_params["rate"]
        scale = 1 / rate
        return [], {"scale": scale}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For Exponential(rate=λ), \mathbb{E}|X-Y| = 2/λ.
        """
        rate = self._bc_params["rate"]
        energy_arr = 2 / rate
        if energy_arr.ndim > 0:
            energy_arr = energy_arr.sum(axis=1)
        return energy_arr

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        Closed form for \mathbb{E}|X - x| with X ~ Exp(rate=λ):
        - if x < 0: 1/λ - x
        - if x >= 0: x - 1/λ + 2 e^{-λ x}/λ
        """
        rate = self._bc_params["rate"]
        # piecewise formula, vectorized
        energy_arr = (x >= 0) * (x - 1 / rate + 2 * np.exp(-rate * x) / rate)
        energy_arr += (x < 0) * (1 / rate - x)
        if energy_arr.ndim > 0:
            energy_arr = energy_arr.sum(axis=1)
        return energy_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the distribution."""
        params1 = {"rate": [1, 2, 2.5, 3.5, 5]}
        params2 = {"rate": 2}
        params3 = {
            "rate": [
                [2, 2, 2],
                [4, 4, 4],
            ],
            "index": pd.Index([1, 2]),
            "columns": pd.Index(["a", "b", "c"]),
        }

        return [params1, params2, params3]
