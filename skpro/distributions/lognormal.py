# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Log-Normal probability distribution."""

__author__ = ["fkiraly"]

from math import exp

import numpy as np
import pandas as pd

from scipy.special import erf, erfinv

from skpro.distributions.base import BaseDistribution


class LogNormal(BaseDistribution):
    """Log-Normal distribution (skpro native).

    Parameters
    ----------
    mean : float or array of float (1D or 2D)
        mean of the normal distribution of the variable's natural logarithm
    sd : float or array of float (1D or 2D), must be positive
        standard deviation of the normal distribution of the logarithm of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.lognormal import LogNormal

    >>> n = LogNormal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)
    """

    _tags = {
        "capabilities:approx": ["pdflognorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.index = index
        self.columns = columns

        # todo: untangle index handling
        # and broadcast of parameters.
        # move this functionality to the base class
        self._mu, self._sigma = self._get_bc_params()
        shape = self._mu.shape

        if index is None:
            index = pd.RangeIndex(shape[0])

        if columns is None:
            columns = pd.RangeIndex(shape[1])

        super(LogNormal, self).__init__(index=index, columns=columns)

    def _get_bc_params(self):
        """Fully broadcast parameters of self, given param shapes and index, columns."""
        to_broadcast = [self.mu, self.sigma]
        if hasattr(self, "index") and self.index is not None:
            to_broadcast += [self.index.to_numpy().reshape(-1, 1)]
        if hasattr(self, "columns") and self.columns is not None:
            to_broadcast += [self.columns.to_numpy()]
        bc = np.broadcast_arrays(*to_broadcast)
        return bc[0], bc[1]

    def energy(self, x=None):
        r"""Energy of self, w.r.t. self or a constant frame x.

        Let :math:`X, Y` be i.i.d. random variables with the distribution of `self`.

        If `x` is `None`, returns :math:`\mathbb{E}[|X-Y|]` (per row), "self-energy".
        If `x` is passed, returns :math:`\mathbb{E}[|X-x|]-0.5\mathbb{E}[|X-Y|]`
        (per row), "CRPS wrt x".

        Parameters
        ----------
        x : None or pd.DataFrame, optional, default=None
            if pd.DataFrame, must have same rows and columns as `self`

        Returns
        -------
        pd.DataFrame with same rows as `self`, single column `"energy"`
        each row contains one float, self-energy/energy as described above.
        """
        approx_spl_size = self.get_tag("approx_energy_spl")
        approx_method = (
            "by approximating the energy expectation by the arithmetic mean of "
            f"{approx_spl_size} samples"
        )

        # splx, sply = i.i.d. samples of X - Y of size N = approx_spl_size
        N = approx_spl_size
        if x is None:
            splx = self.sample(N)
            sply = self.sample(N)
            # approx E[abs(X-Y)] via mean of samples of abs(X-Y) obtained from splx,sply
            spl = splx - sply
            energy = spl.apply(np.linalg.norm, axis=1, ord=1).groupby(level=1).mean()
            energy = pd.DataFrame(energy, index=self.index, columns=["energy"])
        else:
            d = self.loc[x.index, x.columns]
            mu_arr, sd_arr = d._mu, d._sigma
            c_arr = x * (2 * self.cdf(x) - 1)
            c_arr2 = -2 * exp((mu_arr + sd_arr**2) / 2)
            c_arr3 = self.cdf((np.log(x) - mu_arr - sd_arr**2) / sd_arr)
            c_arr3 = c_arr3 + self.cdf(sd_arr / mu_arr**0.5) - 1
            c_arr2 = c_arr2 * c_arr3
            c_arr = c_arr + c_arr2

            energy_arr = np.sum(c_arr, axis=1)
            energy = pd.DataFrame(energy_arr, index=self.index, columns=["energy"])
        return energy

    def mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        mean_arr = np.exp(self._mu + self._sigma**2 / 2)
        return pd.DataFrame(mean_arr, index=self.index, columns=self.columns)

    def var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns :math:`\mathbb{V}[X] = \mathbb{E}\left(X - \mathbb{E}[X]\right)^2`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        mu = self._mu
        sigma = self._sigma
        sd_arr = exp(2 * mu + 2 * sigma**2) - exp(2 * mu + sigma**2)
        return pd.DataFrame(sd_arr, index=self.index, columns=self.columns) ** 2

    def pdf(self, x):
        """Probability density function."""
        d = self.loc[x.index, x.columns]
        pdf_arr = np.exp(-0.5 * ((np.log(x.values) - d.mu) / d.sigma) ** 2)
        pdf_arr = pdf_arr / (x.values * d.sigma * np.sqrt(2 * np.pi))
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def log_pdf(self, x):
        """Logarithmic probability density function."""
        d = self.loc[x.index, x.columns]
        lpdf_arr = -0.5 * ((np.log(x.values) - d.mu) / d.sigma) ** 2
        lpdf_arr = lpdf_arr - np.log(x.values * d.sigma * np.sqrt(2 * np.pi))
        return pd.DataFrame(lpdf_arr, index=x.index, columns=x.columns)

    def cdf(self, x):
        """Cumulative distribution function."""
        d = self.loc[x.index, x.columns]
        cdf_arr = 0.5 + 0.5 * erf((np.log(x.values) - d.mu) / (d.sigma * np.sqrt(2)))
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        d = self.loc[p.index, p.columns]
        icdf_arr = d.mu + d.sigma * np.sqrt(2) * erfinv(2 * p.values - 1)
        icdf_arr = np.exp(self._sigma * icdf_arr)
        return pd.DataFrame(icdf_arr, index=p.index, columns=p.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "sigma": 1}
        params2 = {
            "mu": 0,
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
