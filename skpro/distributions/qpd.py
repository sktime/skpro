"""Johnson Quantile-Parameterized Distributions."""

from __future__ import annotations

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = [
    "FelixWick",
    "setoguchi-naoki",
]  # interface only. Cyclic boosting authors in cyclic_boosting package

import typing
import warnings
from typing import Sequence

if typing.TYPE_CHECKING:
    from cyclic_boosting.quantile_matching import J_QPD_S, J_QPD_B
    from pandas import DataFrame, Index

import numpy as np
import pandas as pd
from scipy.stats import logistic, norm

from skpro.distributions.base import BaseDistribution


class QPD_S(BaseDistribution):
    """Johnson Quantile-Parameterized Distributions with semi-bounded mode.

    see https://repositories.lib.utexas.edu/bitstream/handle/2152
        /63037/HADLOCK-DISSERTATION-2017.pdf
    (Due to the Python keyword, the parameter lambda from
    this reference is named kappa below.)
    A distribution is parameterized by a symmetric-percentile triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float or array_like[float]
        quantile function value of ``alpha``
    qv_median : float or array_like[float]
        quantile function value of quantile 0.5
    qv_high : float or array_like[float]
        quantile function value of quantile ``1 - alpha``
    lower : float
        lower bound of semi-bounded range
    version: str
        options are ``normal`` (default) or ``logistic``

    Example
    -------
    >>> from skpro.distributions.qpd import QPD_S  # doctest: +SKIP

    >>> qpd = QPD_S(
    ...         alpha=0.2,
    ...         qv_low=[1, 2],
    ...         qv_median=[3, 4],
    ...         qv_high=[5, 6],
    ...         lower=0
    ...       )  # doctest: +SKIP

    >>> qpd.mean()  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["setoguchi-naoki", "felix-wick"],
        "maintainers": ["setoguchi-naoki"],
        "python_dependencies": "cyclic_boosting>=1.4.0; findiff",
        # estimator tags
        # --------------
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: float | Sequence,
        qv_median: float | Sequence,
        qv_high: float | Sequence,
        lower: float,
        version: str | None = "normal",
        index=None,
        columns=None,
    ):
        self.qpd = []
        self.alpha = alpha
        self.qv_low = qv_low
        self.qv_median = qv_median
        self.qv_high = qv_high
        self.lower = lower
        self.version = version
        self.index = index
        self.columns = columns

        super().__init__(index=index, columns=columns)

        from cyclic_boosting.quantile_matching import J_QPD_S

        params = [alpha, qv_low, qv_median, qv_high]
        for idx, p in enumerate(params):
            if isinstance(p, float):
                params[idx] = np.array([p])
            elif isinstance(p, (tuple, list, np.ndarray)):
                params[idx] = np.array(p)
            else:
                raise ValueError("data type is not float or array_like object")

        alpha, qv_low, qv_median, qv_high = params[:]
        if index is None:
            index = pd.RangeIndex(qv_low.shape[0])
            self.index = index

        if columns is None:
            columns = pd.RangeIndex(1)
            self.columns = columns

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        if (np.any(qv_low > qv_median)) or np.any(qv_high < qv_median):
            warnings.warn(
                "The SPT values are not monotonically increasing, "
                "each SPT is sorted by value",
                stacklevel=2,
            )
            idx = np.where((qv_low > qv_median), True, False) + np.where(
                (qv_high < qv_median), True, False
            )
            un_orderd_idx = np.argwhere(idx > 0).tolist()
            warnings.warn(f"sorted index {un_orderd_idx}", stacklevel=2)
            for idx in un_orderd_idx:
                low, mid, high = sorted([qv_low[idx], qv_median[idx], qv_high[idx]])
                qv_low[idx] = low
                qv_median[idx] = mid
                qv_high[idx] = high

        self.qpd = J_QPD_S(
            alpha=alpha,
            qv_low=qv_low,
            qv_median=qv_median,
            qv_high=qv_high,
            l=self.lower,
            version=version,
        )

    def mean(self, lower: float = None, upper: float = None):
        """Return expected value of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        if not lower:
            lower = self.lower
        if not upper:
            upper = 1e3
        loc = exp_func(lower, upper, self.qpd, self.index.shape[0])
        return pd.DataFrame(loc, index=self.index, columns=self.columns)

    def var(self, lower: float = None, upper: float = None):
        """Return element/entry-wise variance of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        if not lower:
            lower = self.lower
        if not upper:
            upper = 1e3
        mean = self.mean(lower, upper).values
        var = var_func(mean, lower, upper, self.qpd, self.index.shape[0])
        return pd.DataFrame(var, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        """Probability density function.

        this fucntion transform cdf to pdf
        because j-qpd's pdf calculation is bit complex
        """
        pdf = pdf_func(x, self.qpd, self.index)
        return pd.DataFrame(pdf, index=x.index, columns=x.columns)

    def ppf(self, p: pd.DataFrame):
        """Quantile function = percent point function = inverse cdf."""
        ppf = ppf_func(p, self.qpd, self.index)
        return pd.DataFrame(ppf, index=p.index, columns=p.columns)

    def cdf(self, x: pd.DataFrame):
        """Cumulative distribution function."""
        cdf = cdf_func(x, self.qpd, self.index)
        return pd.DataFrame(cdf, index=x.index, columns=x.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": -0.3,
            "qv_median": 0.0,
            "qv_high": 0.3,
            "lower": -0.5,
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [-0.3, -0.3, -0.3],
            "qv_median": [0.0, 0.0, 0.0],
            "qv_high": [0.3, 0.3, 0.3],
            "lower": -0.5,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a"]),
        }
        return [params1, params2]


class QPD_B(BaseDistribution):
    """Johnson Quantile-Parameterized Distributions with bounded mode.

    see https://repositories.lib.utexas.edu/bitstream/handle/2152
        /63037/HADLOCK-DISSERTATION-2017.pdf
    (Due to the Python keyword, the parameter lambda from
    this reference is named kappa below).
    A distribution is parameterized by a symmetric-percentile triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float or array_like[float]
        quantile function value of ``alpha``
    qv_median : float or array_like[float]
        quantile function value of quantile 0.5
    qv_high : float or array_like[float]
        quantile function value of quantile ``1 - alpha``
    lower : float
        lower bound of semi-bounded range
    upper : float
        upper bound of supported range
    version: str
        options are ``normal`` (default) or ``logistic``

    Example
    -------
    >>> from skpro.distributions.qpd import QPD_B  # doctest: +SKIP

    >>> qpd = QPD_B(
    ...         alpha=0.2,
    ...         qv_low=[1, 2],
    ...         qv_median=[3, 4],
    ...         qv_high=[5, 6],
    ...         lower=0,
    ...         upper=10
    ...       )  # doctest: +SKIP

    >>> qpd.mean()  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["setoguchi-naoki", "felix-wick"],
        "maintainers": ["setoguchi-naoki"],
        "python_dependencies": "cyclic_boosting>=1.2.5",
        # estimator tags
        # --------------
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: float | Sequence,
        qv_median: float | Sequence,
        qv_high: float | Sequence,
        lower: float,
        upper: float,
        version: str | None = "normal",
        index=None,
        columns=None,
    ):
        self.qpd = []
        self.alpha = alpha
        self.qv_low = qv_low
        self.qv_median = qv_median
        self.qv_high = qv_high
        self.lower = lower
        self.upper = upper
        self.version = version
        self.index = index
        self.columns = columns

        super().__init__(index=index, columns=columns)

        from cyclic_boosting.quantile_matching import J_QPD_B

        params = [alpha, qv_low, qv_median, qv_high]
        for idx, p in enumerate(params):
            if isinstance(p, float):
                params[idx] = np.array([p])
            elif isinstance(p, (tuple, list, np.ndarray)):
                params[idx] = np.array(p)
            else:
                raise ValueError("data type is not float or array_like object")

        alpha, qv_low, qv_median, qv_high = params[:]
        if index is None:
            index = pd.RangeIndex(qv_low.shape[0])
            self.index = index

        if columns is None:
            columns = pd.RangeIndex(1)
            self.columns = columns

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        if (np.any(qv_low > qv_median)) or np.any(qv_high < qv_median):
            warnings.warn(
                "The SPT values are not monotonically increasing, "
                "each SPT is sorted by value",
                stacklevel=2,
            )
            idx = np.where((qv_low > qv_median), True, False) + np.where(
                (qv_high < qv_median), True, False
            )
            un_orderd_idx = np.argwhere(idx > 0).tolist()
            warnings.warn(f"sorted index {un_orderd_idx}", stacklevel=2)
            for idx in un_orderd_idx:
                low, mid, high = sorted([qv_low[idx], qv_median[idx], qv_high[idx]])
                qv_low[idx] = low
                qv_median[idx] = mid
                qv_high[idx] = high

        self.qpd = J_QPD_B(
            alpha=alpha,
            qv_low=qv_low,
            qv_median=qv_median,
            qv_high=qv_high,
            l=self.lower,
            u=self.upper,
            version=version,
        )

    def mean(self, lower: float = None, upper: float = None):
        """Return expected value of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        if not lower:
            lower = self.lower
        if not upper:
            upper = self.upper
        loc = exp_func(lower, upper, self.qpd, self.index.shape[0])
        return pd.DataFrame(loc, index=self.index, columns=self.columns)

    def var(self, lower: float = None, upper: float = None):
        """Return element/entry-wise variance of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        if not lower:
            lower = self.lower
        if not upper:
            upper = self.upper
        mean = self.mean(lower, upper).values
        var = var_func(mean, lower, upper, self.qpd, self.index.shape[0])
        return pd.DataFrame(var, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        """Probability density function.

        this fucntion transform cdf to pdf
        because j-qpd's pdf calculation is bit complex
        """
        pdf = pdf_func(x, self.qpd, self.index)
        return pd.DataFrame(pdf, index=x.index, columns=x.columns)

    def ppf(self, p: pd.DataFrame):
        """Quantile function = percent point function = inverse cdf."""
        ppf = ppf_func(p, self.qpd, self.index)
        return pd.DataFrame(ppf, index=p.index, columns=p.columns)

    def cdf(self, x: pd.DataFrame):
        """Cumulative distribution function."""
        cdf = cdf_func(x, self.qpd, self.index)
        return pd.DataFrame(cdf, index=x.index, columns=x.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": -0.3,
            "qv_median": 0.0,
            "qv_high": 0.3,
            "lower": -0.5,
            "upper": 0.5,
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [-0.3, -0.3, -0.3],
            "qv_median": [0.0, 0.0, 0.0],
            "qv_high": [0.3, 0.3, 0.3],
            "lower": -0.5,
            "upper": 0.5,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a"]),
        }
        return [params1, params2]


def calc_pdf(x: np.ndarray, qpd: J_QPD_S | J_QPD_B) -> np.ndarray:
    """Return pdf value for all samples."""
    from findiff import FinDiff

    dx = x[1] - x[0]
    derivative = FinDiff(1, dx, 1)
    cdf = qpd.cdf(x).T
    if cdf.ndim < 2:
        cdf = cdf[np.newaxis, :]
    pdf = np.asarray(derivative(cdf))
    return pdf


def exp_func(lower: float, upper: float, qpd: J_QPD_S | J_QPD_B, size: int):
    """Return Expectation."""
    x = np.linspace(lower, upper, num=int(1e3))
    pdf_arr = calc_pdf(x, qpd)
    x = np.tile(x, (size, 1))
    loc_arr = np.trapz(x * pdf_arr, x, dx=1e-6, axis=1)
    return loc_arr


def var_func(
    mu: np.ndarray, lower: float, upper: float, qpd: J_QPD_S | J_QPD_B, size: int
):
    """Return Variance."""
    x = np.linspace(lower, upper, num=int(1e3))
    pdf_arr = calc_pdf(x, qpd)
    x = np.tile(x, (size, 1))
    var_arr = np.trapz(((x - mu) ** 2) * pdf_arr, x, dx=1e-6, axis=1)
    return var_arr


def pdf_func(x: DataFrame, qpd: J_QPD_S | J_QPD_B, index: Index):
    """Return pdf value."""
    x_value = np.unique(x.values)
    pdf = np.zeros((x.index.shape[0], len(x.columns)))
    for v in x_value:
        x0 = np.linspace(v, v + 1e-3, num=3)
        pdf_arr = calc_pdf(x0, qpd)[:, 0]
        if pdf_arr.ndim < 1:
            pdf_arr = pdf_arr[np.newaxis]
        rows, cols = np.where(x.values == v)
        for r, c in zip(rows, cols):
            id = x.index[r]
            target = index.get_loc(id)
            pdf[r][c] = pdf_arr[target]
    return pdf


def ppf_func(x: DataFrame, qpd: J_QPD_S | J_QPD_B, index: Index):
    """Return ppf value."""
    quantiles = np.unique(x.values)
    ppf = np.zeros((x.index.shape[0], len(x.columns)))
    for q in quantiles:
        ppf_arr = qpd.ppf(q).T
        if ppf_arr.ndim < 1:
            ppf_arr = ppf_arr[np.newaxis]
        rows, cols = np.where(x.values == q)
        for r, c in zip(rows, cols):
            id = x.index[r]
            target = index.get_loc(id)
            ppf[r][c] = ppf_arr[target]
    return pd.DataFrame(ppf, index=x.index, columns=x.columns)


def cdf_func(x: DataFrame, qpd: J_QPD_S | J_QPD_B, index: Index):
    """Return cdf value."""
    x_value = np.unique(x.values)
    cdf = np.zeros((x.index.shape[0], len(x.columns)))
    for v in x_value:
        cdf_arr = qpd.cdf(v).T
        if cdf_arr.ndim < 1:
            cdf_arr = cdf_arr[np.newaxis]
        rows, cols = np.where(x.values == v)
        for r, c in zip(rows, cols):
            id = x.index[r]
            target = index.get_loc(id)
            cdf[r][c] = cdf_arr[target]
    return pd.DataFrame(cdf, index=x.index, columns=x.columns)
