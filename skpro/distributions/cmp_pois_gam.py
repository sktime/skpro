# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Compound poisson gamma probability distribution."""

__author__ = ["ShreeshaM07"]

import math

import numpy as np
from scipy.special import gamma as gam_fun
from scipy.stats import poisson

from skpro.distributions.base import BaseDistribution


class CmpPoissonGamma(BaseDistribution):
    """Compound Poisson Gamma Distribution.

    Parameters
    ----------
    lambda_ : float or array of float (1D or 2D)
        The rate parameter of the Poisson distribution.
    alpha : float or array of float (1D or 2D)
        The shape parameter of the Gamma distribution.
    beta : float or array of float (1D or 2D)
        The rate parameter (inverse scale) of the Gamma distribution.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": [
            "mean",
            "var",
            "energy",
            "pdf",
            "log_pdf",
            "cdf",
            "ppf",
            "pmf",
            "log_pmf",
        ],
        "distr:measuretype": "mixed",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, lambda_, alpha, beta, index=None, columns=None):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        super().__init__(index=index, columns=columns)

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
        lam = self.lambda_
        alpha = self.alpha
        beta = self.beta
        pdf_value = np.zeros_like(x)
        tol = 1e-10

        for idx, val in np.ndenumerate(x):
            if val <= 0:
                continue  # PDF is zero for non-positive values

            const_term = np.exp(-beta * val) / ((np.exp(lam) - 1) * val)
            i = 1
            while True:
                t1 = lam * (pow(beta * val, alpha))
                numer = pow(t1, i)
                i_fact = math.factorial(i)
                gamma_fun = gam_fun(i * alpha)
                denom = i_fact * gamma_fun

                term = numer / denom
                pdf_value[idx] += term

                if term < tol:
                    break
                i += 1
                if i > 1000:  # safeguard to prevent infinite loop
                    break
            pdf_value[idx] = pdf_value[idx] * const_term

        return pdf_value

    def _pmf(self, k):
        """Probability mass function.

        Parameters
        ----------
        k : 2D np.ndarray, same shape as ``self``
            values to evaluate the pmf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pmf values at the given points
        """
        lambda_ = self.lambda_
        return poisson.pmf(k, lambda_)
