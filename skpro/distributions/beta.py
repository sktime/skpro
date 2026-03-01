# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Beta probability distribution."""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import beta, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Beta(_ScipyAdapter):
    r"""Beta distribution.

    Most methods wrap ``scipy.stats.beta``.

    The Beta distribution is parametrized by two shape parameters :math:`\alpha`
    and :math:`\beta`, such that the probability density function (PDF) is given by:

    .. math:: f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}

    where :math:`B(\alpha, \beta)` is the beta function. The beta function
    is a normalization constant to ensure that the total probability is 1.

    Parameters
    ----------
    alpha : float or array of float (1D or 2D), must be positive
    beta : float or array of float (1D or 2D), must be positive
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.beta import Beta

    >>> d = Beta(beta=[[1, 1], [2, 3], [4, 5]], alpha=2)

    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["malikrafsan"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": [
            "mean",
            "var",
            "pdf",
            "log_pdf",
            "cdf",
            "ppf",
            "energy",
        ],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, alpha, beta, index=None, columns=None):
        self.alpha = alpha
        self.beta = beta

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return beta

    def _get_scipy_param(self):
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        return [], {"a": alpha, "b": beta}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        Deterministic quadrature: \mathbb{E}|X-Y| = 2 \int_0^1 F(t)(1-F(t)) dt.
        """
        alpha = self._bc_params["alpha"]
        beta_param = self._bc_params["beta"]

        def self_energy_cell(a, b):
            cdf = lambda t: beta.cdf(t, a=a, b=b)  # noqa: E731
            integral, _ = quad(lambda t: cdf(t) * (1 - cdf(t)), 0, 1, limit=200)
            return 2 * integral

        vec_energy = np.vectorize(self_energy_cell)
        energy_arr = vec_energy(alpha, beta_param)
        if np.ndim(energy_arr) > 1:
            energy_arr = energy_arr.sum(axis=1)
        return energy_arr

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        Uses \mathbb{E}|X - x| = \mathbb{E}[X] - x + 2 \int_0^{x} F(t) dt.
        """
        alpha = self._bc_params["alpha"]
        beta_param = self._bc_params["beta"]
        mean = alpha / (alpha + beta_param)

        def energy_cell(a, b, m, xi):
            if xi <= 0:
                return m - xi
            if xi >= 1:
                # Use mean - xi + 2*(1 - 0) = mean - xi + 2
                # since integral from 0 to 1 of CDF
                cdf = lambda t: beta.cdf(t, a=a, b=b)  # noqa: E731
                integral, _ = quad(cdf, 0, 1, limit=200)
                return m - xi + 2 * integral
            cdf = lambda t: beta.cdf(t, a=a, b=b)  # noqa: E731
            integral, _ = quad(cdf, 0, xi, limit=200)
            return m - xi + 2 * integral

        vec_energy = np.vectorize(energy_cell)
        energy_arr = vec_energy(alpha, beta_param, mean, x)
        if np.ndim(energy_arr) > 1:
            energy_arr = energy_arr.sum(axis=1)
        return energy_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"alpha": [[1, 1], [2, 3], [4, 5]], "beta": 3}
        params2 = {
            "alpha": 2,
            "beta": 3,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"alpha": 1.5, "beta": 2.1}

        return [params1, params2, params3]
