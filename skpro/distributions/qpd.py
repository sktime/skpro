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


class J_QPD_S(BaseDistribution):
    """Johnson Quantile-Parameterized Distributions

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
        self.alpha = alpha
        self.version = version
        self.qv_low = qv_low
        self.qv_median = qv_median
        self.qv_high = qv_high
        self.index = index
        self.columns = columns

        for qv in [qv_low, qv_median, qv_high]:
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

        self.l = l
        self.c = self.phi.ppf(1 - alpha)

        self.L = log(qv_low - l)
        self.H = log(qv_high - l)
        self.B = log(qv_median - l)

        self.n = np.zeros_like(self.L)
        self.theta = qv_median - l

        pos = np.where((self.L + self.H - 2 * self.B) > 0, True, False)
        self.n[pos] = 1
        self.theta[pos] = qv_low[pos] - l

        neg = np.where((self.L + self.H - 2 * self.B) < 0, True, False)
        self.n[neg] = -1
        self.theta[neg] = qv_high[neg] - l

        B_L = self.B - self.L
        H_B = self.H - self.B
        _min = np.where(B_L < H_B, B_L, H_B)

        self.delta = 1.0 / self.c * sinh(arccosh((self.H - self.L) / (2 * _min)))
        self.kappa = 1.0 / (self.delta * self.c) * _min

        # DataFrame
        self.L = pd.DataFrame(self.L, index=index)
        self.H = pd.DataFrame(self.H, index=index)
        self.B = pd.DataFrame(self.B, index=index)
        self.n = pd.DataFrame(self.n, index=index)
        self.theta = pd.DataFrame(self.theta, index=index)
        self.delta = pd.DataFrame(self.delta, index=index)
        self.kappa = pd.DataFrame(self.kappa, index=index)

        super().__init__(index=index, columns=columns)

    def mean(self):
        def cdf_func(x, theta, kappa, delta, n):
            cdf_arr = self.phi.cdf(
                1.0
                / delta
                * sinh(
                    arcsinh(1.0 / kappa * log((x - self.l) / theta))
                    - arcsinh(n * self.c * delta)
                )
            )
            return cdf_arr

        def exp_func(x, theta, kappa, delta, n):
            # TODO: scipy.integrate will be removed in scipy 1.12.0
            pdf = derivative(cdf_func, x, dx=1e-6, args=(theta, kappa, delta, n))
            return x * pdf

        loc = []
        for i in self.index:
            theta = self.theta.loc[i, :].to_numpy()
            kappa = self.kappa.loc[i, :].to_numpy()
            delta = self.delta.loc[i, :].to_numpy()
            n = self.n.loc[i, :].to_numpy()
            # NOTE: integral interval should be checked, -inf to inf will be NaN
            l, _ = quad(exp_func, args=(theta, kappa, delta, n), a=0.0, b=np.inf)
            loc.append(l)

        loc_arr = np.array(loc)
        return pd.DataFrame(loc_arr, index=self.index, columns=self.columns)

    def var(self):
        def cdf_func(x, theta, kappa, delta, n):
            cdf_arr = self.phi.cdf(
                1.0
                / delta
                * sinh(
                    arcsinh(1.0 / kappa * log((x - self.l) / theta))
                    - arcsinh(n * self.c * delta)
                )
            )
            return cdf_arr

        def var_func(x, mu, theta, kappa, delta, n):
            # TODO: scipy.integrate will be removed in scipy 1.12.0
            pdf = derivative(cdf_func, x, dx=1e-6, args=(theta, kappa, delta, n))
            return ((x - mu) ** 2) * pdf

        mean = self.mean()
        var = []
        for i in self.index:
            theta = self.theta.loc[i, :].to_numpy()
            kappa = self.kappa.loc[i, :].to_numpy()
            delta = self.delta.loc[i, :].to_numpy()
            n = self.n.loc[i, :].to_numpy()
            mu = mean.loc[i, :].to_numpy()
            # NOTE: integral interval should be checked, -inf to inf will be NaN
            l, _ = quad(var_func, args=(mu, theta, kappa, delta, n), a=0.0, b=np.inf)
            var.append(l)

        var_arr = np.array(var)
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x):
        # cdf -> pdf calculation because j-qpd's pdf calculation is bit complex
        def cdf_func(x, theta, kappa, delta, n):
            cdf_arr = self.phi.cdf(
                1.0
                / delta
                * sinh(
                    arcsinh(1.0 / kappa * log((x - self.l) / theta))
                    - arcsinh(n * self.c * delta)
                )
            )
            return cdf_arr

        pdf = []
        for i in x.index:
            theta = self.theta.loc[i, :].to_numpy()
            kappa = self.kappa.loc[i, :].to_numpy()
            delta = self.delta.loc[i, :].to_numpy()
            n = self.n.loc[i, :].to_numpy()

            p = derivative(cdf_func, x, dx=1e-6, args=(theta, kappa, delta, n))
            pdf.append(p)

        pdf_arr = np.array(pdf)
        return pd.DataFrame(pdf_arr, index=x.index, columns=["pdf"])

    def ppf(self, p):
        theta = self.theta.loc[p.index, :].to_numpy()
        kappa = self.kappa.loc[p.index, :].to_numpy()
        delta = self.delta.loc[p.index, :].to_numpy()
        n = self.n.loc[p.index, :].to_numpy()
        ppf_arr = self.l + theta * exp(
            kappa * sinh(arcsinh(delta * self.phi.ppf(p)) + arcsinh(n * self.c * delta))
        )
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)

    def cdf(self, x):
        theta = self.theta.loc[x.index, :].to_numpy()
        kappa = self.kappa.loc[x.index, :].to_numpy()
        delta = self.delta.loc[x.index, :].to_numpy()
        n = self.n.loc[x.index, :].to_numpy()
        cdf_arr = self.phi.cdf(
            1.0
            / delta
            * sinh(
                arcsinh(1.0 / kappa * log((x - self.l) / theta))
                - arcsinh(n * self.c * delta)
            )
        )
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)
