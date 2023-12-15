"""Johnson Quantile-Parameterized Distributions."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = [
    "FelixWick",
    "setoguchi-naoki",
]  # interface only. Cyclic boosting authors in cyclic_boosting package

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from cyclic_boosting.quantile_matching import J_QPD_B, J_QPD_S
from scipy.integrate import quad
from scipy.misc import derivative
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
    l : float
        lower bound of semi-bounded range (default is 0)
    version: str
        options are ``normal`` (default) or ``logistic``

    Example
    -------
    >>> from skpro.distributions.qpd import QPD_S

    >>> qpd = QPD_S(
    ...         lower=0.2,
    ...         qv_low=[[1, 2], [3, 4]],
    ...         qv_median=[[3, 4], [5, 6]],
    ...         qv_high=[[5, 6], [7, 8]],
    ...         l=0
    ...       )
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: float or object,
        qv_median: float or object,
        qv_high: float or object,
        l: Optional[float] = 0,
        version: Optional[str] = "normal",
        index=None,
        columns=None,
    ):
        self.qpd = []
        self.alpha = alpha
        self.version = version
        self.qv_low = qv_low
        self.qv_median = qv_median
        self.qv_high = qv_high
        self.l_ = l
        self.index = index
        self.columns = columns

        for qv in [alpha, qv_low, qv_median, qv_high]:
            if isinstance(qv, float):
                qv = np.array([qv])
            elif (
                isinstance(qv, tuple)
                or isinstance(qv, list)
                or isinstance(qv, np.ndarray)
            ):
                qv = np.array(qv)
            else:
                raise ValueError("data type is not float or array_like object")

        shape = self.qv_low.shape
        if index is None:
            self.index = index
            index = pd.RangeIndex(shape[0])

        if columns is None:
            self.columns = columns
            columns = pd.RangeIndex(shape[1])

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        if (np.any(qv_low > qv_median)) or np.any(qv_high < qv_median):
            warnings.warn(
                "The SPT values are not monotonically increasing, "
                "each SPT will be replaced by mean value",
                stacklevel=2,
            )
            idx = np.where((qv_low > qv_median), True, False) + np.where(
                (qv_high < qv_median), True, False
            )
            warnings.warn(
                f"replaced index by mean {np.argwhere(idx > 0).tolist()}", stacklevel=2
            )
            qv_low[idx] = np.nanmean(qv_low)
            qv_median[idx] = np.nanmean(qv_median)
            qv_high[idx] = np.nanmean(qv_high)

        iter = np.nditer(qv_low, flags=["c_index"])
        for _i in iter:
            jqpd = J_QPD_S(
                alpha=alpha,
                qv_low=qv_low[iter.index],
                qv_median=qv_median[iter.index],
                qv_high=qv_high[iter.index],
                l=self.l_,
                version=version,
            )
            self.qpd.append(jqpd)
        self.qpd = pd.DataFrame(self.qpd, index=index)

        super().__init__(index=index, columns=columns)

    def mean(self, lower=0.0, upper=np.inf):
        """Return expected value of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        loc = []
        for idx in self.index:
            qpd = self.qpd.loc[idx, :].values[0]
            l, _ = quad(exp_func, args=(qpd), a=lower, b=upper)
            loc.append(l)
        loc_arr = np.array(loc)
        return pd.DataFrame(loc_arr, index=self.index, columns=self.columns)

    def var(self, lower=0.0, upper=np.inf):
        """Return element/entry-wise variance of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        mean = self.mean()
        var = []
        for idx in self.index:
            mu = mean.loc[idx, :].to_numpy()
            qpd = self.qpd.loc[idx, :].values[0]
            l, _ = quad(var_func, args=(mu, qpd), a=lower, b=upper)
            var.append(l)
        var_arr = np.array(var)
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        """Probability density function.

        this fucntion transform cdf to pdf
        because j-qpd's pdf calculation is bit complex
        """
        pdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _x = x.loc[idx, :]
            _pdf = [derivative(qpd.cdf, x0, dx=1e-6) for x0 in _x]
            pdf.append(_pdf)
        pdf_arr = np.array(pdf)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p: pd.DataFrame):
        """Quantile function = percent point function = inverse cdf."""
        ppf = []
        for idx in p.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _ppf = qpd.ppf(p.loc[idx, :])
            ppf.append(_ppf)
        ppf_arr = np.array(ppf)
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)

    def cdf(self, x: pd.DataFrame):
        """Cumulative distribution function."""
        cdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _cdf = qpd.cdf(x.loc[idx, :])
            cdf.append(_cdf)
        cdf_arr = np.array(cdf)
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

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
            "alpha": [0.2, 0.2, 0.2],
            "version": "normal",
            "qv_low": [0.2, 0.2, 0.2],
            "qv_median": [0.5, 0.5, 0.5],
            "qv_high": [0.8, 0.8, 0.8],
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
    l : float
        lower bound of semi-bounded range
    u : float
        upper bound of supported range
    version: str
        options are ``normal`` (default) or ``logistic``
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: float or object,
        qv_median: float or object,
        qv_high: float or object,
        l: float,
        u: float,
        version: Optional[str] = "normal",
        index=None,
        columns=None,
    ):
        self.qpd = []
        self.alpha = alpha
        self.version = version
        self.qv_low = qv_low
        self.qv_median = qv_median
        self.qv_high = qv_high
        self.l_ = l
        self.u = u
        self.index = index
        self.columns = columns

        for qv in [alpha, qv_low, qv_median, qv_high]:
            if isinstance(qv, float):
                qv = np.array([qv])
            elif (
                isinstance(qv, tuple)
                or isinstance(qv, list)
                or isinstance(qv, np.ndarray)
            ):
                qv = np.array(qv)
            else:
                raise ValueError("data type is not float or array_like object")

        shape = self.qv_low.shape
        if index is None:
            self.index = index
            index = pd.RangeIndex(shape[0])

        if columns is None:
            self.columns = columns
            columns = pd.RangeIndex(shape[1])

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        if (np.any(qv_low > qv_median)) or np.any(qv_high < qv_median):
            warnings.warn(
                "The SPT values are not monotonically increasing, "
                "each SPT will be replaced by mean value",
                stacklevel=2,
            )
            idx = np.where((qv_low > qv_median), True, False) + np.where(
                (qv_high < qv_median), True, False
            )
            warnings.warn(
                f"replaced index by mean {np.argwhere(idx > 0).tolist()}", stacklevel=2
            )
            qv_low[idx] = np.nanmean(qv_low)
            qv_median[idx] = np.nanmean(qv_median)
            qv_high[idx] = np.nanmean(qv_high)

        iter = np.nditer(qv_low, flags=["c_index"])
        for _i in iter:
            jqpd = J_QPD_B(
                alpha=alpha,
                qv_low=qv_low[iter.index],
                qv_median=qv_median[iter.index],
                qv_high=qv_high[iter.index],
                l=self.l_,
                u=self.u,
                version=version,
            )
            self.qpd.append(jqpd)
        self.qpd = pd.DataFrame(self.qpd, index=index)

        super().__init__(index=index, columns=columns)

    def mean(self, lower=0.0, upper=np.inf):
        """Return expected value of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        loc = []
        for idx in self.index:
            qpd = self.qpd.loc[idx, :].values[0]
            l, _ = quad(exp_func, args=(qpd), a=lower, b=upper)
            loc.append(l)
        loc_arr = np.array(loc)
        return pd.DataFrame(loc_arr, index=self.index, columns=self.columns)

    def var(self, lower=0.0, upper=np.inf):
        """Return element/entry-wise variance of the distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        mean = self.mean()
        var = []
        for idx in self.index:
            mu = mean.loc[idx, :].to_numpy()
            qpd = self.qpd.loc[idx, :].values[0]
            l, _ = quad(var_func, args=(mu, qpd), a=lower, b=upper)
            var.append(l)
        var_arr = np.array(var)
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        """Probability density function.

        this fucntion transform cdf to pdf
        because j-qpd's pdf calculation is bit complex
        """
        pdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _x = x.loc[idx, :]
            _pdf = [derivative(qpd.cdf, x0, dx=1e-6) for x0 in _x]
            pdf.append(_pdf)
        pdf_arr = np.array(pdf)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p: pd.DataFrame):
        """Quantile function = percent point function = inverse cdf."""
        ppf = []
        for idx in p.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _ppf = qpd.ppf(p.loc[idx, :])
            ppf.append(_ppf)
        ppf_arr = np.array(ppf)
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)

    def cdf(self, x: pd.DataFrame):
        """Cumulative distribution function."""
        cdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _cdf = qpd.cdf(x.loc[idx, :])
            cdf.append(_cdf)
        cdf_arr = np.array(cdf)
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": 0.2,
            "qv_median": 0.5,
            "qv_high": 0.8,
            "l": 0.0,
            "u": 1.0,
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [0.2, 0.2, 0.2],
            "qv_median": [0.5, 0.5, 0.5],
            "qv_high": [0.8, 0.8, 0.8],
            "l": 0.0,
            "u": 1.0,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a"]),
        }
        return [params1, params2]


def exp_func(x, qpd):
    """Return Expectation."""
    # TODO: scipy.integrate will be removed in scipy 1.12.0
    pdf = derivative(qpd.cdf, x, dx=1e-6)
    return x * pdf


def var_func(x, mu, qpd):
    """Return Variance."""
    # TODO: scipy.integrate will be removed in scipy 1.12.0
    pdf = derivative(qpd.cdf, x, dx=1e-6)
    return ((x - mu) ** 2) * pdf
