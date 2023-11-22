"""J-QPD probability distribution."""

__author__ = ["FelixWick", "setoguchi-naoki"]

import pandas as pd
import numpy as np
import warnings

from numpy import exp, log, sinh, arcsinh, arccosh
from scipy.stats import norm, logistic
from scipy.misc import derivative
from scipy.integrate import quad

from typing import Optional

from skpro.distributions.base import BaseDistribution
from cyclic_boosting.quantile_matching import J_QPD_S, J_QPD_B


class QPD_S(BaseDistribution):
    """Johnson Quantile-Parameterized Distributions with semi-bounded mode

    see https://repositories.lib.utexas.edu/bitstream/handle/2152/63037/HADLOCK-DISSERTATION-2017.pdf
    (Due to the Python keyword, the parameter lambda from this reference is named kappa below.).
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
        self.l = l
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

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        if (np.any(qv_low > qv_median)) or np.any((qv_high < qv_median)):
            warnings.warn(
                "The SPT values are not monotonically increasing, each SPT will be replaced by mean value"
            )
            idx = np.where((qv_low > qv_median), True, False) + np.where(
                (qv_high < qv_median), True, False
            )
            print(f"replaced index by mean {np.argwhere(idx > 0).tolist()}")
            qv_low[idx] = np.nanmean(qv_low)
            qv_median[idx] = np.nanmean(qv_median)
            qv_high[idx] = np.nanmean(qv_high)

        iter = np.nditer(qv_low, flags=["c_index"])
        for x in iter:
            jqpd = J_QPD_S(
                alpha=alpha,
                qv_low=qv_low[iter.index],
                qv_median=qv_median[iter.index],
                qv_high=qv_high[iter.index],
                l=l,
                version=version,
            )
            self.qpd.append(jqpd)
        self.qpd = pd.DataFrame(self.qpd, index=index)

        super().__init__(index=index, columns=columns)

    def mean(self, lower=0.0, upper=np.inf):
        def exp_func(x, qpd):
            # TODO: scipy.integrate will be removed in scipy 1.12.0
            pdf = derivative(qpd.cdf, x, dx=1e-6)
            return x * pdf

        loc = []
        for idx in self.index:
            qpd = self.qpd.loc[idx, :].values[0]
            # NOTE: integral interval should be checked, -inf to inf will be NaN
            l, _ = quad(exp_func, args=(qpd), a=lower, b=upper)
            loc.append(l)
        loc_arr = np.array(loc)
        return pd.DataFrame(loc_arr, index=self.index, columns=self.columns)

    def var(self, lower=0.0, upper=np.inf):
        def var_func(x, mu, qpd):
            # TODO: scipy.integrate will be removed in scipy 1.12.0
            pdf = derivative(qpd.cdf, x, dx=1e-6)
            return ((x - mu) ** 2) * pdf

        mean = self.mean()
        var = []
        for idx in self.index:
            mu = mean.loc[idx, :].to_numpy()
            qpd = self.qpd.loc[idx, :].values[0]
            # NOTE: integral interval should be checked, -inf to inf will be NaN
            l, _ = quad(var_func, args=(mu, qpd), a=lower, b=upper)
            var.append(l)
        var_arr = np.array(var)
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        # cdf -> pdf calculation because j-qpd's pdf calculation is bit complex
        pdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _x = x.loc[idx, :]
            _pdf = [derivative(qpd.cdf, x0, dx=1e-6) for x0 in _x]
            pdf.append(_pdf)
        pdf_arr = np.array(pdf)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p: pd.DataFrame):
        ppf = []
        for idx in p.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _ppf = qpd.ppf(p.loc[idx, :])
            ppf.append(_ppf)
        ppf_arr = np.array(ppf)
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)

    def cdf(self, x: pd.DataFrame):
        cdf_arr = np.empty_like(x.shape)
        cdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _cdf = qpd.cdf(x.loc[idx, :])
            cdf.append(_cdf)
        cdf_arr = np.array(cdf)
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)


class QPD_B(BaseDistribution):
    """Johnson Quantile-Parameterized Distributions with bounded mode

    see https://repositories.lib.utexas.edu/bitstream/handle/2152/63037/HADLOCK-DISSERTATION-2017.pdf
    (Due to the Python keyword, the parameter lambda from this reference is named kappa below.).
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
        self.l = l
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

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        if (np.any(qv_low > qv_median)) or np.any((qv_high < qv_median)):
            warnings.warn(
                "The SPT values are not monotonically increasing, each SPT will be replaced by mean value"
            )
            idx = np.where((qv_low > qv_median), True, False) + np.where(
                (qv_high < qv_median), True, False
            )
            print(f"replaced index by mean {np.argwhere(idx > 0).tolist()}")
            qv_low[idx] = np.nanmean(qv_low)
            qv_median[idx] = np.nanmean(qv_median)
            qv_high[idx] = np.nanmean(qv_high)

        iter = np.nditer(qv_low, flags=["c_index"])
        for x in iter:
            jqpd = J_QPD_B(
                alpha=alpha,
                qv_low=qv_low[iter.index],
                qv_median=qv_median[iter.index],
                qv_high=qv_high[iter.index],
                l=l,
                u=u,
                version=version,
            )
            self.qpd.append(jqpd)
        self.qpd = pd.DataFrame(self.qpd, index=index)

        super().__init__(index=index, columns=columns)

    def mean(self, lower=0.0, upper=np.inf):
        def exp_func(x, qpd):
            # TODO: scipy.integrate will be removed in scipy 1.12.0
            pdf = derivative(qpd.cdf, x, dx=1e-6)
            return x * pdf

        loc = []
        for idx in self.index:
            qpd = self.qpd.loc[idx, :].values[0]
            # NOTE: integral interval should be checked, -inf to inf will be NaN
            l, _ = quad(exp_func, args=(qpd), a=lower, b=upper)
            loc.append(l)
        loc_arr = np.array(loc)
        return pd.DataFrame(loc_arr, index=self.index, columns=self.columns)

    def var(self, lower=0.0, upper=np.inf):
        def var_func(x, mu, qpd):
            # TODO: scipy.integrate will be removed in scipy 1.12.0
            pdf = derivative(qpd.cdf, x, dx=1e-6)
            return ((x - mu) ** 2) * pdf

        mean = self.mean()
        var = []
        for idx in self.index:
            mu = mean.loc[idx, :].to_numpy()
            qpd = self.qpd.loc[idx, :].values[0]
            # NOTE: integral interval should be checked, -inf to inf will be NaN
            l, _ = quad(var_func, args=(mu, qpd), a=lower, b=upper)
            var.append(l)
        var_arr = np.array(var)
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x: pd.DataFrame):
        # cdf -> pdf calculation because j-qpd's pdf calculation is bit complex
        pdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _x = x.loc[idx, :]
            _pdf = [derivative(qpd.cdf, x0, dx=1e-6) for x0 in _x]
            pdf.append(_pdf)
        pdf_arr = np.array(pdf)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p: pd.DataFrame):
        ppf = []
        for idx in p.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _ppf = qpd.ppf(p.loc[idx, :])
            ppf.append(_ppf)
        ppf_arr = np.array(ppf)
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)

    def cdf(self, x: pd.DataFrame):
        cdf_arr = np.empty_like(x.shape)
        cdf = []
        for idx in x.index:
            qpd = self.qpd.loc[idx, :].values[0]
            _cdf = qpd.cdf(x.loc[idx, :])
            cdf.append(_cdf)
        cdf_arr = np.array(cdf)
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)
