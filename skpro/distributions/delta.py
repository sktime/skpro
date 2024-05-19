# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Delta (constant/certain) probability distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class Delta(BaseDistribution):
    r"""Delta distribution aka constant distribution aka certain distribution.

    This distribution always produces the same value when sampling - ``c``.
    It it useful to represent a constant value as a distribution, e.g., as a baseline
    method to create a probabilistic prediction from a point prediction.

    The delta distribution is parametrized by a constant value :math:`c`.
    For the cdf, we have:

    .. math:: F(x) = 0 \text{ if } x < c, 1 \text{ if } x \geq c

    The constant value :math:`c` is represented by the parameter ``c``.

    Parameters
    ----------
    c : float or array of float (1D or 2D)
        support of the delta distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.delta import Delta

    >>> delta = Delta(c=[[0, 1], [2, 3], [4, 5]])
    >>> this_is_always_c = delta.sample()
    """

    _tags = {
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "energy", "pmf", "log_pmf", "cdf", "ppf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "nonparametric",
        "broadcast_init": "on",
    }

    def __init__(self, c, index=None, columns=None):
        self.c = c
        if index is None and hasattr(c, "index") and isinstance(c.index, pd.Index):
            index = c.index
        if columns is None and hasattr(c, "columns"):
            if isinstance(c.columns, pd.Index):
                columns = c.columns

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
        # energy of self w.r.t. self is always 0
        energy_arr = self._coerce_to_self_index_np(0)
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
        c = self._bc_params["c"]

        energy_arr = np.abs(c - x)
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
        return self._bc_params["c"]

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        # variance of a constant is always 0
        return self._coerce_to_self_index_np(0)

    def _pmf(self, x):
        """Probability mass function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pmf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pmf values at the given points
        """
        c = self._bc_params["c"]
        pmf_arr = np.where(x == c, 1, 0)
        return pmf_arr

    def _log_pmf(self, x):
        """Logarithmic probability mass function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pmf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            log pmf values at the given points
        """
        c = self._bc_params["c"]
        lpmf_arr = np.where(x == c, 0, -np.inf)
        return lpmf_arr

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
        c = self._bc_params["c"]

        cdf_arr = np.where(x < c, 0, 1)
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
        c = self._bc_params["c"]
        icdf_arr = c
        return icdf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"c": [[0, 1], [2, 3], [4, 5]]}
        params2 = {
            "c": 42,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"c": 42}
        return [params1, params2, params3]
