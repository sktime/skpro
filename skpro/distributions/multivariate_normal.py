# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Multivariate Normal/Gaussian probability distribution."""

__author__ = ["bhavikar04", "fkiraly"]

import numpy as np
import pandas as pd
from scipy.special import erf, erfinv
from scipy.stats import multivariate_normal

from skpro.distributions.base import BaseDistribution
class MultivariateNormal(BaseDistribution):
    r"""Multivariate Normal distribution (skpro native).

    This distribution is multivariate, without correlation between dimensions
    for the array-valued case.

    The multivariate normal distribution is parametrized by mean :math:`\mu` and
    variance-covariance matrix :math:`\Sigma`, such that the pdf is

    .. math:: f(x) = \frac{1} { (|\Sigma|^{\frac{1}{2}}) (2\pi)^{\frac{p}{2}} } \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right) \quad \ # noqa E501


    The mean :math:`\mu` is represented by the parameter ``mu``,
    and the variance-covariance matrix :math:`\Sigma` by the parameter ``cov``.

    Parameters
    ----------
    mu : array of float (px1)
        population mean vector 
    cov : array of float (2D)
        variance-covariance matrix
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.normal import Multivariate_Normal

    >>> n = Multivariate_Normal(mu=[[0, 1], [2, 3], [4, 5]], cov=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self.mu = mu
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
        

        return energy_arr"""

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
        """mu = self._bc_params["mu"]
        cov = self._bc_params["cov"]

        cdf = self.cdf(x)
        pdf = self.pdf(x)
       
        return energy_arr """

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        return self._bc_params["mu"]

    def _var_covar_matrix(self):
        r"""Return variance covariance matrix of the distribution.

        Returns
        -------
        2D np.ndarray, square matrix with dimensions equal to size of mean vector
           
        """
        return self._bc_params["cov"]

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
        cov = self._bc_params["cov"]
        pdf_arr = (
            1 / ((np.linalg.det(cov))**(1/2) * (2*np.pi)**(len(x)/2))
        ) * np.exp(-0.5 * (x - mu).T @ np.linalg.inv(cov) @ (x - mu).cov())  
    
        return pdf_arr

    def _log_pdf(self, x):
        """Logarithmic probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            log pdf values at the given points
        """
        mu = self._bc_params["mu"]
        cov = self._bc_params["cov"]

        lpdf_arr = -0.5 * [
            np.log(np.linalg.det(cov))
            +(x - mu).T @ np.linalg.inv(cov) @ (x - mu)
            +(len(x)/2)*np.log(2*np.pi)
        ]
        
        return lpdf_arr

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
        mu = self._bc_params["mu"]
        cov = self._bc_params["cov"]
        
        cdf_arr= multivariate_normal.cdf(x, mu, cov, allow_singular=False)
        
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
    #Does not exist since closed form of cdf does not exist

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "cov": 1}
        params2 = {
            "mu": 0,
            "cov": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": 1, "cov": 2}
        return [params1, params2, params3]
