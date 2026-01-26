# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Zero-Inflated Negative Binomial probability distribution."""

__author__ = ["marrov"]

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import nbinom, rv_discrete

from skpro.distributions.adapters.scipy import _ScipyAdapter


class ZINB(_ScipyAdapter):
    r"""Zero-Inflated Negative Binomial distribution.

    The Zero-Inflated Negative Binomial distribution models count data with
    excess zeros. It is a mixture of a point mass at zero and a Negative Binomial
    distribution.

    Mathematically:

    .. math::
        P(X = x) = \begin{cases}
            \pi + (1-\pi) \cdot P_{NB}(0) & \text{if } x = 0 \\
            (1-\pi) \cdot P_{NB}(x) & \text{if } x > 0
        \end{cases}

    where :math:`\pi` is the probability of excess zeros, and :math:`P_{NB}(x)`
    is the Negative Binomial probability mass function with mean :math:`\mu`
    and dispersion :math:`\alpha`.

    Parameters
    ----------
    mu : ArrayLike
        Mean of the underlying Negative Binomial distribution.
    alpha : ArrayLike
        Dispersion parameter of the underlying Negative Binomial distribution.
        Higher alpha means more overdispersion.
    pi : ArrayLike
        Probability of excess zeros (zero-inflation probability).
        Must be in [0, 1).
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions import ZINB

    >>> distr = ZINB(mu=2.0, alpha=1.0, pi=0.3)
    >>> distr.mean()
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pmf", "log_pmf", "cdf", "ppf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(
        self,
        mu: ArrayLike,
        alpha: ArrayLike,
        pi: ArrayLike,
        index=None,
        columns=None,
    ):
        self.mu = mu
        self.alpha = alpha
        self.pi = pi

        super().__init__(index=index, columns=columns)

        # Validate parameters after broadcasting
        mu_arr = self._bc_params["mu"]
        alpha_arr = self._bc_params["alpha"]
        pi_arr = self._bc_params["pi"]

        if np.any(mu_arr <= 0):
            raise ValueError("mu must be positive (> 0)")
        if np.any(alpha_arr <= 0):
            raise ValueError("alpha must be positive (> 0)")
        if np.any(pi_arr < 0) or np.any(pi_arr >= 1):
            raise ValueError("pi must be in [0, 1)")

    def _get_scipy_object(self) -> rv_discrete:
        return nbinom

    def _get_scipy_param(self) -> dict:
        """Get scipy parameters for the underlying Negative Binomial."""
        mu = self._bc_params["mu"]
        alpha = self._bc_params["alpha"]

        # Convert to scipy's n, p parametrization
        n = alpha
        p = alpha / (alpha + mu)

        return [n, p], {}

    def _pmf(self, x):
        """Return probability mass function evaluated at x."""
        pi = self._bc_params["pi"]
        args, kwds = self._get_scipy_param()
        nb_pmf = nbinom.pmf(x, *args, **kwds)

        is_zero = x == 0
        result = np.where(
            is_zero,
            pi + (1 - pi) * nb_pmf,
            (1 - pi) * nb_pmf,
        )

        # Negative values have zero probability
        return np.where(x < 0, 0.0, result)

    def _log_pmf(self, x):
        """Return log probability mass function evaluated at x."""
        pi = self._bc_params["pi"]
        args, kwds = self._get_scipy_param()

        is_zero = x == 0
        nb_pmf_at_0 = nbinom.pmf(0, *args, **kwds)
        nb_log_pmf = nbinom.logpmf(x, *args, **kwds)

        # For x=0: log(pi + (1-pi)*P_NB(0))
        log_prob_zero = np.log(pi + (1 - pi) * nb_pmf_at_0)
        # For x>0: log(1-pi) + log(P_NB(x))
        log_prob_positive = np.log(1 - pi) + nb_log_pmf

        result = np.where(is_zero, log_prob_zero, log_prob_positive)

        # Negative values have -inf log probability
        return np.where(x < 0, -np.inf, result)

    def _cdf(self, x):
        """Return cumulative distribution function evaluated at x."""
        pi = self._bc_params["pi"]
        args, kwds = self._get_scipy_param()
        nb_cdf = nbinom.cdf(x, *args, **kwds)

        # CDF: pi + (1-pi) * F_NB(x) for x >= 0
        result = pi + (1 - pi) * nb_cdf

        # Negative values have zero CDF
        return np.where(x < 0, 0.0, result)

    def _ppf(self, p):
        """Return percent point function (inverse CDF) evaluated at p."""
        pi = self._bc_params["pi"]
        args, kwds = self._get_scipy_param()

        # Total probability at 0
        nb_pmf_at_0 = nbinom.pmf(0, *args, **kwds)
        prob_zero = pi + (1 - pi) * nb_pmf_at_0

        # For p <= prob_zero, return 0
        # For p > prob_zero, invert the NB CDF
        # p = pi + (1-pi) * F_NB(x)
        # => F_NB(x) = (p - pi) / (1 - pi)
        q_rescaled = (p - pi) / (1 - pi)
        q_rescaled = np.clip(q_rescaled, 0.0, 1.0)

        y_positive = nbinom.ppf(q_rescaled, *args, **kwds)

        return np.where(p <= prob_zero, 0.0, y_positive)

    def _mean(self):
        """Return mean of the distribution."""
        mu = self._bc_params["mu"]
        pi = self._bc_params["pi"]

        # E[X] = (1-pi) * E[NB] = (1-pi) * mu
        return (1 - pi) * mu

    def _var(self):
        """Return variance of the distribution."""
        mu = self._bc_params["mu"]
        alpha = self._bc_params["alpha"]
        pi = self._bc_params["pi"]

        # Var(NB) = mu + mu^2/alpha
        nb_var = mu + mu**2 / alpha

        # Var(X) = (1-pi) * (Var_NB + mu^2*pi)
        # Using the formula for zero-inflated variance:
        # Var(X) = (1-pi) * Var_NB + (1-pi) * pi * mu^2
        return (1 - pi) * nb_var + (1 - pi) * pi * mu**2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"mu": [[1, 1], [2, 3], [4, 5]], "alpha": 2.0, "pi": 0.3}
        params2 = {
            "mu": 2.0,
            "alpha": 1.0,
            "pi": 0.2,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        params3 = {"mu": 5.0, "alpha": 3.0, "pi": 0.5}
        return [params1, params2, params3]
