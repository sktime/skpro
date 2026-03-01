# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Rayleigh probability distribution."""

__author__ = ["KaranSinghDev"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class Rayleigh(BaseDistribution):
    r"""Rayleigh distribution.

    The Rayleigh distribution is a continuous probability distribution for
    non-negative-valued random variables. It is often observed when the
    overall magnitude of a vector is related to its directional components.

    The Rayleigh distribution is parametrized by a scale parameter :math:`\sigma`,
    such that the pdf is given by:

    .. math:: f(x) = \frac{x}{\sigma^2} e^{-x^2 / (2\sigma^2)}

    The scale parameter :math:`\sigma` is represented by the parameter ``scale``.

    Parameters
    ----------
    scale : float or array of float (1D or 2D), default=1.0
        The scale parameter of the distribution (sigma). Must be positive.
    index : pd.Index, optional, default=None
        Index for the distribution.
    columns : pd.Index, optional, default=None
        Columns for the distribution.

    Examples
    --------
    >>> from skpro.distributions.rayleigh import Rayleigh
    >>> dist = Rayleigh(scale=[[1.0],[2.0]])
    """

    _tags = {
        "authors": ["KaranSinghDev"],
        "maintainers": ["KaranSinghDev"],
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
        "broadcast_init": "on",
    }

    def __init__(self, scale=1.0, index=None, columns=None):
        self.scale = scale
        super().__init__(index=index, columns=columns)

    def _mean(self):
        r"""Return expected value of the distribution.

        For Rayleigh distribution, expectation is given by:
        :math:`\sigma \sqrt{\frac{\pi}{2}}`
        """
        scale = self._bc_params["scale"]
        return scale * np.sqrt(np.pi / 2)

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        For Rayleigh distribution, variance is given by:
        :math:`\frac{4 - \pi}{2} \sigma^2`
        """
        scale = self._bc_params["scale"]
        return ((4 - np.pi) / 2) * scale**2

    def _pdf(self, x):
        """Probability density function."""
        scale = self._bc_params["scale"]

        # PDF formula: (x / sigma^2) * exp(-x^2 / 2sigma^2) for x >= 0
        pdf_arr = (x / scale**2) * np.exp(-0.5 * (x / scale) ** 2)
        pdf_arr = pdf_arr * (x >= 0)
        return pdf_arr

    def _log_pdf(self, x):
        """Logarithmic probability density function."""
        scale = self._bc_params["scale"]

        # Log PDF: log(x) - 2log(sigma) - x^2 / 2sigma^2
        lpdf_arr = np.log(x) - 2 * np.log(scale) - 0.5 * (x / scale) ** 2

        # Handle x < 0 (return -inf) and x=0 (log(0) is -inf)
        lpdf_arr = np.where(x > 0, lpdf_arr, -np.inf)
        return lpdf_arr

    def _cdf(self, x):
        """Cumulative distribution function."""
        scale = self._bc_params["scale"]

        # CDF: 1 - exp(-x^2 / 2sigma^2) for x >= 0
        cdf_arr = 1 - np.exp(-0.5 * (x / scale) ** 2)
        cdf_arr = cdf_arr * (x >= 0)
        return cdf_arr

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        scale = self._bc_params["scale"]

        # PPF: sigma * sqrt(-2 * ln(1 - p))
        # If p is within [0, 1] (base class usually handles this, but for safety)
        ppf_arr = scale * np.sqrt(-2 * np.log(1 - p))
        return ppf_arr

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For Rayleigh(sigma), E[|X-Y|] = 2 * sigma * (sqrt(pi) - sqrt(2)).
        This is a closed-form solution derived from the specific energy integral.
        """
        scale = self._bc_params["scale"]
        energy_arr = scale * np.sqrt(np.pi) * (np.sqrt(2) - 1)
        if hasattr(energy_arr, "ndim") and energy_arr.ndim > 1:
            energy_arr = energy_arr.sum(axis=1)
        return energy_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"scale": [1.0, 2.0]}
        params2 = {"scale": 0.5}
        params3 = {
            "scale": [[1.0, 1.0], [2.0, 3.0]],
            "index": pd.Index([1, 2]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2, params3]
