"""Test class for default methods.

This is not for direct use, but for testing whether the defaulting in various
methods works.

Testing works via TestAllDistributions which discovers the classes in
here, executes the public methods in interface conformance tests,
which in turn triggers the fallback defaults.
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
from scipy.special import erfinv

from skpro.distributions.base import BaseDistribution
from skpro.utils.estimator_checks import check_estimator


# normal distribution with exact implementations removed
class _DistrDefaultMethodTester(BaseDistribution):
    """Tester distribution for default methods."""

    _tags = {
        "capabilities:approx": ["pdfnorm", "mean", "var", "energy", "log_pdf", "cdf"],
        "capabilities:exact": ["pdf", "ppf"],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma

        super().__init__(index=index, columns=columns)

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Parameters
        ----------
        p : 2D np.ndarray, same shape as ``self``
            values to evaluate the ppf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            ppf values at the given points
        """
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        icdf_arr = mu + sigma * np.sqrt(2) * erfinv(2 * p - 1)
        return icdf_arr

    def _pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pdf values at the given points
        """
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        pdf_arr = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        pdf_arr = pdf_arr / (sigma * np.sqrt(2 * np.pi))
        return pdf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "sigma": 1}
        params2 = {
            "mu": 0,
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": 1, "sigma": 2}
        return [params1, params2, params3]


def test_base_default():
    """Test default methods.

    The _DistributionDefaultMethodTester class is not detected
    by TestAllDistributions (it is private), so we need to test it explicitly.

    check_estimator invokes a TestAllDistributions call.
    """
    check_estimator(_DistrDefaultMethodTester, raise_exceptions=True)
