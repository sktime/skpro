# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Beta probability distribution."""

import numpy as np
import pandas as pd
from scipy.special import betaln
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
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": [
            "mean",
            "var",
            "pdf",
            "log_pdf",
            "cdf",
            "ppf",
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

    def _log_pdf(self, x):
        r"""Logarithm of the probability density function.

        For Beta(alpha, beta), the log-pdf is:

        .. math::
            \log f(x) = (\alpha-1)\log x + (\beta-1)\log(1-x)
                        - \log B(\alpha, \beta)

        for :math:`0 < x < 1`, and :math:`-\infty` otherwise.

        Uses ``scipy.special.betaln`` for numerical stability, avoiding
        ``RuntimeWarning: divide by zero`` from the base class fallback
        ``np.log(self.pdf(x))``.
        """
        alpha = self._bc_params["alpha"]
        beta_param = self._bc_params["beta"]

        in_support = (x > 0) & (x < 1)
        # Compute log-pdf only where in support; use 0.5 as safe fallback value
        x_safe = np.where(in_support, x, 0.5)
        log_pdf_val = (
            (alpha - 1) * np.log(x_safe)
            + (beta_param - 1) * np.log(1 - x_safe)
            - betaln(alpha, beta_param)
        )
        return np.where(in_support, log_pdf_val, -np.inf)

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
