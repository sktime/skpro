# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Triangular probability distribution."""

__author__ = ["Ashish-Kumar-Dash"]
__all__ = ["Triangular"]

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, triang

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Triangular(_ScipyAdapter):
    r"""Triangular distribution.

    Most methods wrap ``scipy.stats.triang``.

    The Triangular distribution is parametrized by lower limit :math:`a`,
    mode :math:`c`, and upper limit :math:`b`, with
    :math:`a \leq c \leq b` and :math:`a < b`, such that the pdf is

    .. math::

        f(x; a, c, b) =
        \begin{cases}
            \dfrac{2(x - a)}{(b - a)(c - a)}, & a \leq x < c \\[1ex]
            \dfrac{2(b - x)}{(b - a)(b - c)}, & c \leq x \leq b \\[1ex]
            0, & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    lower : float or array of float (1D or 2D)
        lower limit of the distribution; must satisfy ``lower < upper``
    mode : float or array of float (1D or 2D)
        mode (peak) of the distribution; must satisfy ``lower <= mode <= upper``
    upper : float or array of float (1D or 2D)
        upper limit of the distribution; must satisfy ``lower < upper``
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions import Triangular

    >>> d = Triangular(lower=0, mode=1, upper=3)
    """

    _tags = {
        "authors": ["Ashish-Kumar-Dash"],
        "capabilities:exact": [
            "mean",
            "var",
            "energy",
            "pdf",
            "log_pdf",
            "cdf",
            "ppf",
        ],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
    }

    def __init__(self, lower, mode, upper, index=None, columns=None):
        self.lower = lower
        self.mode = mode
        self.upper = upper

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return triang

    def _get_scipy_param(self):
        lower = self._bc_params["lower"]
        mode = self._bc_params["mode"]
        upper = self._bc_params["upper"]

        scale = upper - lower
        c = (mode - lower) / scale
        return [c], {"loc": lower, "scale": scale}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        a = self._bc_params["lower"]
        c = self._bc_params["mode"]
        b = self._bc_params["upper"]

        d = b - a
        p = c - a
        q = b - c

        energy_arr = 2 * (p**2 + q**2) / (3 * d) - 2 * (p**3 + q**3) / (
            5 * d**2
        )

        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|]`, where :math:`X` is a copy of self,
        and :math:`x` is a constant.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to compute energy w.r.t. to

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        a = self._bc_params["lower"]
        c = self._bc_params["mode"]
        b = self._bc_params["upper"]

        d = b - a
        p = c - a
        q = b - c
        mu = (a + b + c) / 3

        # guard degenerate mode-at-boundary cases; the guarded branch is never
        # selected there (its region is empty), so the value is discarded
        pp = np.where(p == 0, 1.0, p)
        qq = np.where(q == 0, 1.0, q)

        # I_F(x) = integral of the CDF from a to x, piecewise by region
        i_below = np.zeros_like(x + a * 0.0)
        i_left = (x - a) ** 3 / (3 * d * pp)
        i_mid = (
            p**2 / (3 * d) - q**2 / (3 * d) + (x - c) + (b - x) ** 3 / (3 * d * qq)
        )
        i_above = (b - mu) + (x - b)
        i_f = np.select([x <= a, x <= c, x <= b], [i_below, i_left, i_mid], i_above)

        # E|X - x| = 2 I_F(x) - I_F(b) + (b - x),  with I_F(b) = b - mu
        energy_arr = 2 * i_f - (b - mu) + (b - x)

        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "lower": [[0, -1], [1, 0], [2, 1]],
            "mode": [[1, 0], [2, 1], [3, 2]],
            "upper": [[3, 2], [4, 3], [5, 4]],
        }
        params2 = {
            "lower": 0,
            "mode": 1,
            "upper": 3,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        params3 = {"lower": -2, "mode": 4, "upper": 4}

        return [params1, params2, params3]
