# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Exponential probability distribution."""

__author__ = ["ShreeshaM07"]

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import gamma, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Gamma(_ScipyAdapter):
    r"""Gamma Distribution.

    Most methods wrap ``scipy.stats.gamma``.

    The Gamma Distribution is parameterized by shape :math:`\alpha` and
    rate :math:`\beta`, such that the pdf is

    .. math:: f(x) = \frac{x^{\alpha-1}\exp\left(-\beta x\right) \beta^{\alpha}}{\tau(\alpha)}

    where :math:`\tau(\alpha)` is the Gamma function.
    For all positive integers, :math:`\tau(\alpha) = (\alpha-1)!`.

    Parameters
    ----------
    alpha : float or array of float (1D or 2D)
        It represents the shape parameter.
    beta : float or array of float (1D or 2D)
        It represents the rate parameter which is also
        inverse of the scale parameter.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.gamma import Gamma

    >>> d = Gamma(beta=[[1, 1], [2, 3], [4, 5]], alpha=2)

    Energy computations (exact, via deterministic numerical quadrature):

    >>> d_scalar = Gamma(alpha=2, beta=1)
    >>> d_scalar.energy()  # E|X-Y|
    """  # noqa: E501

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf", "energy"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, alpha, beta, index=None, columns=None):
        self.alpha = alpha
        self.beta = beta

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return gamma

    def _get_scipy_param(self):
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]
        scale = 1 / beta

        return [], {"a": alpha, "scale": scale}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        Uses deterministic 1D quadrature:
        \\mathbb{E}|X-Y| = 4 \\int_0^\\infty F(t)(1-F(t)) dt,
        where F is the Gamma CDF.
        """
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        def self_energy_cell(a, b):
            cdf = lambda t: gamma.cdf(t, a=a, scale=1 / b)  # noqa: E731
            integral, _ = quad(lambda t: cdf(t) * (1 - cdf(t)), 0, np.inf, limit=200)
            return 4 * integral

        vec_energy = np.vectorize(self_energy_cell)
        energy_arr = vec_energy(alpha, beta)
        if np.ndim(energy_arr) > 1:
            energy_arr = energy_arr.sum(axis=1)
        return energy_arr

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        Uses \mathbb{E}|X - x| = \mathbb{E}[X] - x + 2 \int_0^{x} F(t) dt
        (with empty integral if x<0).
        """
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        def energy_cell(a, b, xi):
            if xi <= 0:
                return a / b - xi

            cdf = lambda t: gamma.cdf(t, a=a, scale=1 / b)  # noqa: E731
            integral, _ = quad(cdf, 0, xi, limit=200)
            return a / b - xi + 2 * integral

        vec_energy = np.vectorize(energy_cell)
        energy_arr = vec_energy(alpha, beta, x)
        if np.ndim(energy_arr) > 1:
            energy_arr = energy_arr.sum(axis=1)
        return energy_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"alpha": [6, 2.5], "beta": [[1, 1], [2, 3], [4, 5]]}
        params2 = {
            "alpha": 2,
            "beta": 3,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"alpha": 1.5, "beta": 2.1}

        return [params1, params2, params3]
