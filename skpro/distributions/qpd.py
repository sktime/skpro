"""Johnson Quantile-Parameterized Distributions."""

from __future__ import annotations

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = [
    "FelixWick",
    "setoguchi-naoki",
]  # interface only. Cyclic boosting authors in cyclic_boosting package

import typing
import warnings

if typing.TYPE_CHECKING:
    from typing import Sequence, Optional, Union

    from cyclic_boosting.quantile_matching import J_QPD_S, J_QPD_B
    from pandas import DataFrame, Index

import numpy as np
import pandas as pd
from scipy.stats import logistic, norm

from skpro.distributions.base import BaseDistribution, _DelegatedDistribution


class QPD_Johnson(_DelegatedDistribution):
    """Johnson Quantile-Parameterized Distribution.

    A Johnson QPD distribution is parameterized by a symmetric-percentile triplet (SPT),
    at quantiles alpha, 0.5, and 1-alpha, respectively.

    see https://repositories.lib.utexas.edu/bitstream/handle/2152
        /63037/HADLOCK-DISSERTATION-2017.pdf
    Parameter names are as in the reference, except for the parameter lambda,
    which is renamed to kappa, as lambda is a reserved keyword in python.

    This class allows selection of the mode bounding type,
    i.e. semi-bounded, bounded, or unbounded.

    * if neither ``lower`` nor ``upper`` bound is given, the mode is unbounded
    * if only ``lower`` bound is given, the mode is semi-bounded
    * if both ``lower`` and ``upper`` bounds are given, the mode is bounded

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
    lower : float, default = None (no lower bound)
        lower bound of semi-bounded range or bounded range
    upper : float, default = None (no upper bound)
        upper bound of bounded range
    version: str, one of ``'normal'`` (default), ``'logistic'``
        options are ``'normal'`` (default) or ``'logistic'``
    dist_shape: float, optional, default=0.0
        parameter modifying the logistic base distribution via
        sinh/arcsinh-scaling (only active in sinhlogistic version)

    Example
    -------
    >>> from skpro.distributions.qpd import QPD_Johnson  # doctest: +SKIP

    >>> qpd = QPD_Johnson(
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
        "authors": ["setoguchi-naoki", "felix-wick", "fkiraly"],
        "maintainers": ["setoguchi-naoki"],
        "python_dependencies": "cyclic_boosting>=1.2.5",
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf", "pdf", "log_pdf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: Union[float, Sequence],
        qv_median: Union[float, Sequence],
        qv_high: Union[float, Sequence],
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        version: Optional[str] = "normal",
        dist_shape: Optional[float] = 0.0,
        index=None,
        columns=None,
    ):
        self.alpha = alpha
        self.qv_low = qv_low
        self.qv_median = qv_median
        self.qv_high = qv_high
        self.lower = lower
        self.upper = upper
        self.version = version
        self.dist_shape = dist_shape
        self.index = index
        self.columns = columns

        if lower is None:
            delegate_cls = QPD_U
            extra_params = {"dist_shape": dist_shape}
        elif upper is None:
            delegate_cls = QPD_S
            extra_params = {"lower": lower}
        else:
            delegate_cls = QPD_B
            extra_params = {"lower": lower, "upper": upper}

        params = {
            "alpha": alpha,
            "qv_low": qv_low,
            "qv_median": qv_median,
            "qv_high": qv_high,
            "version": version,
            "index": index,
            "columns": columns,
            **extra_params,
        }

        self.delegate_ = delegate_cls(**params)

        self.index = self.delegate_.index
        self.columns = self.delegate_.columns

        super().__init__(index=self.index, columns=self.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": 0.2,
            "qv_median": 0.5,
            "qv_high": 0.8,
        }
        params2 = {
            "alpha": 0.1,
            "version": "normal",
            "qv_low": [0.2, 0.2, 0.2],
            "qv_median": [0.5, 0.5, 0.5],
            "qv_high": [0.8, 0.8, 0.8],
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a"]),
        }
        params3 = {
            "alpha": 0.1,
            "version": "normal",
            "qv_low": [0.1, 0.2, 0.3],
            "qv_median": [0.4, 0.5, 0.6],
            "qv_high": [0.7, 0.8, 0.9],
            "lower": 0.05,
        }
        params4 = {
            "alpha": 0.12,
            "version": "logistic",
            "qv_low": [0.25, 0.2, 0.22],
            "qv_median": [0.45, 0.51, 0.54],
            "qv_high": [0.85, 0.83, 0.81],
            "lower": 0.05,
            "upper": 0.95,
        }
        return [params1, params2, params3, params4]


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
        "python_dependencies": ["cyclic_boosting>=1.4.0"],
        # estimator tags
        # --------------
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: Union[float, Sequence],
        qv_median: Union[float, Sequence],
        qv_high: Union[float, Sequence],
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

        alpha, qv_low, qv_median, qv_high = _prep_qpd_params(
            self, alpha, qv_low, qv_median, qv_high
        )

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

        Please set the upper and lower limits of the random variable correctly.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        if not lower:
            lower = self.lower
        if not upper:
            upper = 1e3
        x = np.linspace(lower, upper, num=int(1e3))
        cdf_arr = self.qpd.cdf(x).T
        loc = exp_func(x, cdf_arr, self.index.shape[0])
        return pd.DataFrame(loc, index=self.index, columns=self.columns)

    def var(self, lower: float = None, upper: float = None):
        """Return element/entry-wise variance of the distribution.

        Please set the upper and lower limits of the random variable correctly.

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
        x = np.linspace(lower, upper, num=int(1e3))
        cdf_arr = self.qpd.cdf(x).T
        var = var_func(x, mean, cdf_arr, self.index.shape[0])
        return pd.DataFrame(var, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        """Probability density function.

        this fucntion transform cdf to pdf
        because j-qpd's pdf calculation is bit complex
        """
        return pdf_func(x, self.qpd, self.index)

    def ppf(self, p: pd.DataFrame):
        """Quantile function = percent point function = inverse cdf."""
        return ppf_func(p, self.qpd, self.index)

    def cdf(self, x: pd.DataFrame):
        """Cumulative distribution function."""
        return cdf_func(x, self.qpd, self.index)

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
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a"]),
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [-0.3, -0.3, -0.3],
            "qv_median": [0.0, 0.0, 0.0],
            "qv_high": [0.3, 0.3, 0.3],
            "lower": -0.5,
            "index": pd.RangeIndex(3),
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
        "python_dependencies": ["cyclic_boosting>=1.4.0"],
        # estimator tags
        # --------------
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: Union[float, Sequence],
        qv_median: Union[float, Sequence],
        qv_high: Union[float, Sequence],
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

        alpha, qv_low, qv_median, qv_high = _prep_qpd_params(
            self, alpha, qv_low, qv_median, qv_high
        )

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

        Please set the upper and lower limits of the random variable correctly.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        if not lower:
            lower = self.lower
        if not upper:
            upper = self.upper
        x = np.linspace(lower, upper, num=int(1e3))
        cdf_arr = self.qpd.cdf(x).T
        loc = exp_func(x, cdf_arr, self.index.shape[0])
        return pd.DataFrame(loc, index=self.index, columns=self.columns)

    def var(self, lower: float = None, upper: float = None):
        """Return element/entry-wise variance of the distribution.

        Please set the upper and lower limits of the random variable correctly.

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
        x = np.linspace(lower, upper, num=int(1e3))
        cdf_arr = self.qpd.cdf(x).T
        var = var_func(x, mean, cdf_arr, self.index.shape[0])
        return pd.DataFrame(var, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        """Probability density function.

        this fucntion transform cdf to pdf
        because j-qpd's pdf calculation is bit complex
        """
        return pdf_func(x, self.qpd, self.index)

    def ppf(self, p: pd.DataFrame):
        """Quantile function = percent point function = inverse cdf."""
        return ppf_func(p, self.qpd, self.index)

    def cdf(self, x: pd.DataFrame):
        """Cumulative distribution function."""
        return cdf_func(x, self.qpd, self.index)

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
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a"]),
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [-0.3, -0.3, -0.3],
            "qv_median": [0.0, 0.0, 0.0],
            "qv_high": [0.3, 0.3, 0.3],
            "lower": -0.5,
            "upper": 0.5,
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a"]),
        }
        return [params1, params2]


class QPD_U(BaseDistribution):
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
    version: str
        options are ``normal`` (default) or ``logistic``
    dist_shape: str
        parameter modifying the logistic base distribution via
        sinh/arcsinh-scaling (only active in sinhlogistic version)

    Example
    -------
    >>> from skpro.distributions.qpd import QPD_U  # doctest: +SKIP

    >>> qpd = QPD_U(
    ...         alpha=0.2,
    ...         qv_low=[1, 2],
    ...         qv_median=[3, 4],
    ...         qv_high=[5, 6],
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
        qv_low: Union[float, Sequence],
        qv_median: Union[float, Sequence],
        qv_high: Union[float, Sequence],
        version: str | None = "normal",
        dist_shape: Optional[float] = 0.0,
        index=None,
        columns=None,
    ):
        self.qpd = []
        self.alpha = alpha
        self.qv_low = qv_low
        self.qv_median = qv_median
        self.qv_high = qv_high
        self.version = version
        self.dist_shape = dist_shape
        self.index = index
        self.columns = columns

        super().__init__(index=index, columns=columns)

        from cyclic_boosting.quantile_matching import J_QPD_extended_U

        alpha, qv_low, qv_median, qv_high = _prep_qpd_params(
            self, alpha, qv_low, qv_median, qv_high
        )

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

        iter = np.nditer(qv_low, flags=["c_index"])
        for _i in iter:
            jqpd = J_QPD_extended_U(
                alpha=alpha,
                qv_low=qv_low[iter.index],
                qv_median=qv_median[iter.index],
                qv_high=qv_high[iter.index],
                version=version,
                shape=dist_shape,
            )
            self.qpd.append(jqpd)
        self.qpd = pd.DataFrame(self.qpd, index=self.index)

    def mean(self, lower: float = -1e3, upper: float = 1e3):
        """Return expected value of the distribution.

        Please set the upper and lower limits of the random variable correctly.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        cdf_arr = []
        x = np.linspace(lower, upper, num=int(1e3))
        for idx in self.index:
            qpd = self.qpd.loc[idx, :].values[0]
            cdf_arr.append(qpd.cdf(x))
        cdf_arr = np.asarray(cdf_arr)
        loc = exp_func(x, cdf_arr, self.index.shape[0])
        return pd.DataFrame(loc, index=self.index, columns=self.columns)

    def var(self, lower: float = -1e3, upper: float = 1e3):
        """Return element/entry-wise variance of the distribution.

        Please set the upper and lower limits of the random variable correctly.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        mean_arr = self.mean(lower, upper).values
        cdf_arr = []
        x = np.linspace(lower, upper, num=int(1e3))
        for idx in self.index:
            qpd = self.qpd.loc[idx, :].values[0]
            cdf_arr.append(qpd.cdf(x))
        cdf_arr = np.asarray(cdf_arr)
        var_arr = var_func(x, mean_arr, cdf_arr, self.index.shape[0])
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        """Probability density function.

        this fucntion transform cdf to pdf
        because j-qpd's pdf calculation is bit complex
        """
        return pdf_func(x, self.qpd, self.index)

    def ppf(self, p: pd.DataFrame):
        """Quantile function = percent point function = inverse cdf."""
        return ppf_func(p, self.qpd, self.index)

    def cdf(self, x: pd.DataFrame):
        """Cumulative distribution function."""
        return cdf_func(x, self.qpd, self.index)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": 0.2,
            "qv_median": 0.5,
            "qv_high": 0.8,
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a"]),
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [0.2, 0.2, 0.2],
            "qv_median": [0.5, 0.5, 0.5],
            "qv_high": [0.8, 0.8, 0.8],
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a"]),
        }
        return [params1, params2]


def calc_pdf(x: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Return pdf value for all samples."""
    from findiff import FinDiff

    dx = x[1] - x[0]
    derivative = FinDiff(1, dx, 1)
    if cdf.ndim < 2:
        cdf = cdf[np.newaxis, :]
    pdf = np.asarray(derivative(cdf))
    return pdf


def exp_func(x: np.ndarray, cdf: np.ndarray, size: int):
    """Return Expectation."""
    pdf_arr = calc_pdf(x, cdf)
    x = np.tile(x, (size, 1))
    loc = np.trapz(x * pdf_arr, x, dx=1e-6, axis=1)
    return loc


def var_func(x: np.ndarray, mu: np.ndarray, cdf: np.ndarray, size: int):
    """Return Variance."""
    pdf_arr = calc_pdf(x, cdf)
    x = np.tile(x, (size, 1))
    var = np.trapz(((x - mu) ** 2) * pdf_arr, x, dx=1e-6, axis=1)
    return var


def pdf_func(x: DataFrame, dist: J_QPD_S | J_QPD_B | pd.DataFrame, index: Index):
    """Return pdf value."""
    qpd = dist.values if isinstance(dist, pd.DataFrame) else dist
    prob_var = np.unique(x.values)
    pdf = np.zeros((x.index.shape[0], len(x.columns)))
    for v in prob_var:
        # all qpds
        x0 = np.linspace(v, v + 1e-3, num=3)
        if isinstance(dist, pd.DataFrame):
            cdf_arr = np.asarray([func[0].cdf(x0) for func in qpd])
        else:
            cdf_arr = qpd.cdf(x0).T
        pdf_arr = calc_pdf(x0, cdf_arr)[:, 0]
        if pdf_arr.ndim < 1:
            pdf_arr = pdf_arr[np.newaxis]
        # pick up
        rows, cols = np.where(x.values == v)
        for r, c in zip(rows, cols):
            id = x.index[r]
            target = index.get_loc(id)
            pdf[r][c] = pdf_arr[target]
    return pd.DataFrame(pdf, index=x.index, columns=x.columns)


def ppf_func(x: DataFrame, dist: J_QPD_S | J_QPD_B | pd.DataFrame, index: Index):
    """Return ppf value."""
    qpd = dist.values if isinstance(dist, pd.DataFrame) else dist
    quantiles = np.unique(x.values)
    ppf = np.zeros((x.index.shape[0], len(x.columns)))
    for q in quantiles:
        # all qpds
        if isinstance(dist, pd.DataFrame):
            ppf_arr = np.asarray([func[0].ppf(q) for func in qpd])
        else:
            ppf_arr = qpd.ppf(q).T
        if ppf_arr.ndim < 1:
            ppf_arr = ppf_arr[np.newaxis]
        # pick up
        rows, cols = np.where(x.values == q)
        for r, c in zip(rows, cols):
            id = x.index[r]
            target = index.get_loc(id)
            ppf[r][c] = ppf_arr[target]
    return pd.DataFrame(ppf, index=x.index, columns=x.columns)


def cdf_func(x: DataFrame, dist: J_QPD_S | J_QPD_B | pd.DataFrame, index: Index):
    """Return cdf value."""
    qpd = dist.values if isinstance(dist, pd.DataFrame) else dist
    x_value = np.unique(x.values)
    cdf = np.zeros((x.index.shape[0], len(x.columns)))
    for v in x_value:
        # all qpds
        if isinstance(dist, pd.DataFrame):
            cdf_arr = np.asarray([func[0].cdf(v) for func in qpd])
        else:
            cdf_arr = qpd.cdf(v).T
        if cdf_arr.ndim < 1:
            cdf_arr = cdf_arr[np.newaxis]
        # pick up
        rows, cols = np.where(x.values == v)
        for r, c in zip(rows, cols):
            id = x.index[r]
            target = index.get_loc(id)
            cdf[r][c] = cdf_arr[target]
    return pd.DataFrame(cdf, index=x.index, columns=x.columns)


def _prep_qpd_params(self, alpha, qv_low, qv_median, qv_high):
    """Prepare parameters for Johnson Quantile-Parameterized Distributions."""
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha])
    qv_low, qv_median, qv_high = BaseDistribution._get_bc_params(
        self, qv_low, qv_median, qv_high, oned_as="col"
    )
    qv_low = qv_low.flatten()
    qv_median = qv_median.flatten()
    qv_high = qv_high.flatten()
    return alpha, qv_low, qv_median, qv_high
