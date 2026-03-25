# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Exponential probability distribution."""

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

    Parameters
    ----------
    rate : float or array of float (1D or 2D)
        rate of the distribution
        rate = 1/scale
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.exponential import Exponential
    >>> d = Exponential(rate=2)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ShreeshaM07"],
        # estimator tags
        # --------------
        "capabilities:approx": ["ppf", "pdfnorm"],
        "capabilities:exact": [
            "mean",
            "var",
            "pdf",
            "log_pdf",
            "cdf",
            "energy",
            "truncated_mean",
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

        For Exponential(rate=λ), \mathbb{E}|X-Y| = 1/λ.
        """
        rate = self._bc_params["rate"]
        energy_arr = 1 / rate
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

    def _truncated_mean(self, lower, upper):
        r"""Return expected value of the distribution truncated to [lower, upper].

        For :math:`X \sim \text{Exp}(\lambda)` with support :math:`[0, \infty)`:

        .. math::

            \mathbb{E}[X \mid a < X < b]
            = \frac{(a + 1/\lambda) e^{-\lambda a}
                  - (b + 1/\lambda) e^{-\lambda b}}
                   {e^{-\lambda a} - e^{-\lambda b}}

        where :math:`a = \max(\text{lower}, 0)`.

        Parameters
        ----------
        lower : 2D np.ndarray, same shape as ``self``
            lower truncation bound
        upper : 2D np.ndarray, same shape as ``self``
            upper truncation bound

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            truncated expected value of distribution (entry-wise)
        """
        rate = self._bc_params["rate"]
        inv_rate = 1.0 / rate

        a = np.maximum(lower, 0.0)
        b = np.asarray(upper, dtype=float)

        b_safe = np.where(np.isfinite(b), b, 0.0)
        exp_a = np.exp(-rate * a)
        exp_b_raw = np.exp(-rate * b_safe)
        is_b_finite = np.isfinite(b)

        term_a = (a + inv_rate) * exp_a
        term_b = np.where(is_b_finite, (b_safe + inv_rate) * exp_b_raw, 0.0)

        numer = term_a - term_b
        denom = exp_a - np.where(is_b_finite, exp_b_raw, 0.0)

        safe_denom = np.where(np.abs(denom) < 1e-15, np.nan, denom)
        result = numer / safe_denom

        no_overlap = (a >= b) | ((upper <= 0) & np.isfinite(upper))
        result = np.where(no_overlap, np.nan, result)
        return result

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
