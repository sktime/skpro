# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Uniform probability distribution."""

__author__ = ["an20805"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class Uniform(BaseDistribution):
    r"""Continuous uniform distribution.

    The uniform distribution is parameterized by lower and upper bounds of interval,
    :math:`a` and :math`b`, such that the pdf is

    .. math:: f(x) = \frac{1}{b - a} \text{ for } a \leq x \leq b, \text{ and } 0 \text{ otherwise}  # noqa E501

    The lower bound :math:`a` is represented by the parameter ``lower``,
    and the upper bound :math:`b` by the parameter ``upper``.

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
    >>> from skpro.distributions import Uniform

    >>> u = Uniform(lower=0, upper=5)
    """

    _tags = {
        "authors": ["an20805"],
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["pdf", "log_pdf", "cdf", "ppf", "mean", "var", "energy"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, lower, upper, index=None, columns=None):
        self.lower = lower
        self.upper = upper

        super().__init__(index=index, columns=columns)

        if self.ndim == 0 and lower >= upper:
            raise ValueError(
                "Error in Uniform distribution parameters, "
                "upper bound must be strictly greater than "
                "lower bound."
            )
        else:
            # use 2D broadcasted params for checking
            lower = self._bc_params["lower"]
            upper = self._bc_params["upper"]

            if np.any(lower >= upper):
                raise ValueError(
                    "Error in Uniform distribution parameters, "
                    "upper bound must be strictly greater than "
                    "lower bound."
                )

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
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]

        in_bounds = np.logical_and(x >= lower, x <= upper)
        pdf_arr = in_bounds / (upper - lower)
        return pdf_arr

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
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]

        in_bounds = (x >= lower) & (x <= upper)
        above_bound = x > upper

        cdf_arr = in_bounds * (x - lower) / (upper - lower) + above_bound
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
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]

        ppf_arr = lower + p * (upper - lower)
        return ppf_arr

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]

        mean_arr = (lower + upper) / 2
        return mean_arr

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]

        var_arr = (upper - lower) ** 2 / 12
        return var_arr

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Private method, to be implemented by subclasses.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]

        energy_arr = (upper - lower) / 3  # Expected absolute difference

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
        a = self._bc_params["lower"]
        b = self._bc_params["upper"]

        is_outside = np.logical_or(x < a, x > b)
        is_inside = 1 - is_outside

        midpoint = (a + b) / 2
        energy_arr = is_outside * np.abs(x - midpoint)
        energy_arr += is_inside * ((b - x) ** 2 + (a - x) ** 2) / (2 * (b - a))

        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"lower": 0, "upper": [5, 10]}
        params2 = {
            "lower": -5,
            "upper": 5,
            "index": pd.Index([1, 3, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"lower": 0, "upper": 3}

        return [params1, params2, params3]
