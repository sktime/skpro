# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Compound poisson gamma probability distribution."""

__author__ = ["ShreeshaM07"]

import math

import numpy as np

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
        from scipy.special import gamma as gam_fun

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
        from scipy.stats import poisson

        lambda_ = self.lambda_
        return poisson.pmf(k, lambda_)

    def _compute_crj(self, r, j, rho):
        from itertools import combinations_with_replacement

        from scipy.special import factorial

        c_rj = 0
        partitions = [
            p for p in combinations_with_replacement(range(1, r + 1), j) if sum(p) == r
        ]
        for partition in partitions:
            term = 1
            for s_i in partition:
                term *= factorial(rho + 1 + s_i) / (
                    factorial(rho - 1) * factorial(s_i + 2)
                )
            c_rj += term
        c_rj *= (rho**2 + rho) ** (-r / 2 - j)
        return c_rj

    def _compute_hr(self, x, r, rho):
        from scipy.special import eval_hermitenorm, factorial

        hr = np.zeros_like(x, dtype=float)
        for j in range(1, r + 1):
            H = eval_hermitenorm(r + 2 * j, x)
            crj = self._compute_crj(r, j, rho)
            hr += H * crj / factorial(j)
        return hr

    def _cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the cdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            cdf values at the given points
        """
        rho = self.alpha
        max_r = 10  # Adjust as needed
        phi_x = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
        series_sum = np.zeros_like(x, dtype=float)
        for r in range(1, max_r + 1):
            hr_x = self._compute_hr(x, r, rho)
            series_sum += hr_x
            # print(f"r={r}, hr_x={hr_x}, series_sum={series_sum}")
            # if np.any(np.abs(hr_x) < 1e-6):  # Convergence check
            #     break
        cdf = phi_x * (1 + series_sum)
        return cdf
