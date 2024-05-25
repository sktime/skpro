# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Half-Normal probability distribution."""

__author__ = ["SaiRevanth25"]

import numpy as np
import pandas as pd
from scipy.special import erf, erfinv

from skpro.distributions.base import BaseDistribution


class HalfNormal(BaseDistribution):
    r"""Half-Normal distribution (skpro native).

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The half-normal distribution is parametrized by the standard deviation
    :math:`\sigma`, such that the pdf is

    .. math:: f(x) = \frac{\sqrt{2}}{\sigma \sqrt{\pi}}
                    \exp\left(-\frac{x^2}{2\sigma^2}\right)

    The standard deviation :math:`\sigma` is represented by the parameter ``sigma``.

    Parameters
    ----------
    sigma : float or array of float (1D or 2D), must be positive
        standard deviation of the half-normal distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.halfnormal import HalfNormal

    >>> hn = HalfNormal(sigma=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, sigma, index=None, columns=None):
        self.sigma = sigma

        super().__init__(index=index, columns=columns)

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Private method, to be implemented by subclasses.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        sigma = self._bc_params["sigma"]

        energy_arr = sigma * np.sqrt(2 - 2 / np.pi)
        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|]`, where :math:`X` is a copy of self,
        and :math:`x` is a constant.

        Private method, to be implemented by subclasses.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to compute energy w.r.t. to

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        sigma = self._bc_params["sigma"]

        cdf = self.cdf(x)
        pdf = self.pdf(x)

        energy_arr = (x) * (2 * cdf - 1) + 2 * sigma**2 * pdf
        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        sigma = self._bc_params["sigma"]
        mean_arr = sigma * np.sqrt(2 / np.pi)

        return mean_arr

    def _var(self):
        """Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        sigma = self._bc_params["sigma"]
        var_arr = sigma**2 * (1 - 2 / np.pi)
        return var_arr

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
        sigma = self._bc_params["sigma"]

        pdf_arr = (np.sqrt(2 / np.pi) / sigma) * np.exp(-0.5 * (x / sigma) ** 2)
        return pdf_arr

    def _log_pdf(self, x):
        """Logarithmic probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the log pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            log pdf values at the given points
        """
        sigma = self._bc_params["sigma"]
        log_pdf_arr = -0.5 * (x / sigma) ** 2
        log_pdf_arr = log_pdf_arr - np.log(np.sqrt(2 / np.pi) / sigma)
        return log_pdf_arr

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
        sigma = self._bc_params["sigma"]

        cdf_arr = erf(x / (sigma * np.sqrt(2)))
        return cdf_arr

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
        sigma = self._bc_params["sigma"]

        ppf_arr = sigma * np.sqrt(2) * erfinv(p)
        return ppf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"sigma": [[1, 2], [3, 4]]}
        params2 = {
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"sigma": 2}
        return [params1, params2, params3]
