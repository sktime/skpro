# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Laplace probability distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class Laplace(BaseDistribution):
    r"""Laplace distribution.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The Laplace distribution is parametrized by mean :math:`\mu` and
    scale :math:`b`, such that the pdf is

    .. math:: f(x) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)

    The mean :math:`\mu` is represented by the parameter ``mu``,
    and the scale :math:`b` by the parameter ``scale``.

    It should be noted that this parametrization differs from the mean/standard
    deviation parametrization, which is also common in the literature.
    The standard deviation of this distribution is :math:`\sqrt{2} s`.

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        mean of the distribution
    scale : float or array of float (1D or 2D), must be positive
        scale parameter of the distribution, same as standard deviation / sqrt(2)
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions import Laplace

    >>> n = Laplace(mu=[[0, 1], [2, 3], [4, 5]], scale=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, scale, index=None, columns=None):
        self.mu = mu
        self.scale = scale

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
        energy_arr = self._bc_params["scale"]
        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1) * 1.5
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
        mu = self._bc_params["mu"]
        sc = self._bc_params["scale"]

        y_arr = np.abs((x - mu) / sc)
        c_arr = y_arr + np.exp(-y_arr)
        energy_arr = sc * c_arr
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
        return self._bc_params["mu"]

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        sc = self._bc_params["scale"]
        sd_arr = np.sqrt(2) * sc
        return sd_arr**2

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
        sc = self._bc_params["scale"]
        pdf_arr = np.exp(-np.abs((x - mu) / sc))
        pdf_arr = pdf_arr / (2 * sc)
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
        sc = self._bc_params["scale"]
        lpdf_arr = -np.abs((x - mu) / sc)
        lpdf_arr = lpdf_arr - np.log(2 * sc)
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
        sc = self._bc_params["scale"]

        sgn_arr = np.sign(x - mu)
        exp_arr = np.exp(-np.abs((x - mu) / sc))
        cdf_arr = 0.5 + 0.5 * sgn_arr * (1 - exp_arr)
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
        mu = self._bc_params["mu"]
        sc = self._bc_params["scale"]

        sgn_arr = np.sign(p - 0.5)
        icdf_arr = mu - sc * sgn_arr * np.log(1 - 2 * np.abs(p - 0.5))
        return icdf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "scale": 1}
        params2 = {
            "mu": 0,
            "scale": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": -0.5, "scale": 3}

        return [params1, params2, params3]
