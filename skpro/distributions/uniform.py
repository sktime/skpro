# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Uniform probability distribution."""

__author__ = ["an20805"]

import numpy as np
import pandas as pd

from sktime.proba.base import BaseDistribution


class Uniform(BaseDistribution):
    """Continuous uniform distribution.

    Parameters
    ----------
    lower : float
        Lower bound of the distribution.
    upper : float, must be greater than lower
        Upper bound of the distribution.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from sktime.proba import Uniform

    >>> u = Uniform(lower=0, upper=5)
    """

    _tags = {
        "authors": ["an20805"],  
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["pdf", "log_pdf", "cdf", "ppf", "mean", "var"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, lower, upper, index=None, columns=None):
        if lower >= upper:
            raise ValueError(f"Upper bound ({upper}) must be greater than lower bound {lower}.")
        self.lower = lower
        self.upper = upper
        self.index = index
        self.columns = columns

        self._lower, self._upper = self._get_bc_params(self.lower, self.upper)

        if index is None:
            index = pd.RangeIndex(1)  

        if columns is None:
            columns = pd.RangeIndex(1)  

        super().__init__(index=index, columns=columns)

    def pdf(self, x):
        """Probability density function."""
        d = self.loc[x.index, x.columns]
        in_bounds = (x.values >= d.lower) & (x.values <= d.upper)
        pdf_arr = np.where(in_bounds, 1 / (d.upper - d.lower), 0)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def log_pdf(self, x):
        """Logarithmic probability density function."""
        return np.log(self.pdf(x))

    def cdf(self, x):
        """Cumulative distribution function."""
        d = self.loc[x.index, x.columns]
        cdf_arr = np.where(
            x.values < d.lower, 0, np.where(x.values > d.upper, 1, (x.values - d.lower) / (d.upper - d.lower))
        )
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p):
        """Quantile function (inverse CDF)."""
        d = self.loc[p.index, p.columns]
        ppf_arr = d.lower + p.values * (d.upper - d.lower)
        return pd.DataFrame(ppf_arr, index=p.index, columns=d.columns)

    def mean(self):
        """Mean of the distribution."""
        return pd.DataFrame((self._lower + self._upper) / 2, index=self.index, columns=self.columns)
    
    def energy(self, x=None):
        """Energy of self, w.r.t. self or a constant frame x.
        Let X, Y be i.i.d. random variables with the distribution of self.
        If x is None, returns the expected absolute difference between two random variables (self-energy).
        If x is passed, returns the expected absolute difference between a random variable and a constant value.

        Parameters
        ----------
        x : None or pd.DataFrame, optional, default=None
            if pd.DataFrame, must have same rows and columns as `self`

        Returns
        -------
        pd.DataFrame with same rows as `self`, single column `"energy"`
            each row contains one float, self-energy/energy as described above.
        """

        if x is None:
            # Self-energy
            a_arr, b_arr = self._lower, self._upper
            energy_arr = (b_arr - a_arr) / 3  # Expected absolute difference
            energy = pd.DataFrame(energy_arr, index=self.index, columns=["energy"])
        else:
            # Energy wrt constant frame x
            a_arr, b_arr = self._lower, self._upper
            midpoint = (a_arr + b_arr) / 2
            energy_arr = np.where(
                x < a_arr, np.abs(x - midpoint),
                np.where(x > b_arr, np.abs(x - midpoint), ((b_arr - x) ** 2 + (a_arr - x) ** 2) / (2 * (b_arr - a_arr))),
            )
            energy = pd.DataFrame(energy_arr, index=self.index, columns=["energy"])
        return energy


    def var(self):
        """Variance of the distribution."""
        return pd.DataFrame(((self._upper - self._lower) ** 2) / 12, index=self.index, columns=self.columns)
    
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"lower": 0, "upper": 10}
        params2 = {
            "lower": -5,
            "upper": 5, 
            "index": pd.Index([1, 3, 5]), 
            "columns": pd.Index(["a", "b"])
        }
        return [params1, params2]

