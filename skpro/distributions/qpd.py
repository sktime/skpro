"""Johnson Quantile-Parameterized Distributions."""

from __future__ import annotations

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = [
    "FelixWick",
    "setoguchi-naoki",
]  # interface only. Cyclic boosting authors in cyclic_boosting package

from typing import Sequence

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
    lower : float, default = None
        lower bound of bounded range for QPD.
        This is used when estimating QPD and calculating
        expectation and variance
    upper : float, default = None
        upper bound of bounded range for QPD.
        This is used when estimating QPD and calculating
        expectation and variance
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
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf", "pdf"],
        "distr:measuretype": "continuous",
    }

    def __init__(
        self,
        alpha: float,
        qv_low: float | Sequence,
        qv_median: float | Sequence,
        qv_high: float | Sequence,
        lower: float | None = None,
        upper: float | None = None,
        version: str | None = "normal",
        dist_shape: float | None = 0.0,
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
            "qv_low": [[-0.3], [-0.2], [-0.1]],
            "qv_median": [[-0.1], [0.0], [0.1]],
            "qv_high": [[0.2], [0.3], [0.4]],
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
            "qv_low": [0.15, 0.1, 0.15],
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
        lower bound of semi-bounded range.
        This is used when estimating QPD and calculating
        expectation and variance
    upper : float, default = 1e3
        upper bound of probability density function to
        calculate expected value and variance
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
        "authors": ["setoguchi-naoki", "felix-wick", "fkiraly"],
        "maintainers": ["setoguchi-naoki"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf", "pdf"],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
        "broadcast_params": [
            "alpha",
            "qv_low",
            "qv_median",
            "qv_high",
            "lower",
            "upper",
        ],
    }

    def __init__(
        self,
        alpha: float,
        qv_low: float | Sequence,
        qv_median: float | Sequence,
        qv_high: float | Sequence,
        lower: float,
        upper: float = 1e3,
        version: str | None = "normal",
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
        self.index = index
        self.columns = columns

        super().__init__(index=index, columns=columns)

        # precompute parameters for methods
        phi = _resolve_phi(version)
        self.phi = phi

        qpd_params = _prep_qpd_vars(phi=phi, mode="S", **self._bc_params)
        self._qpd_params = qpd_params

    def _ppf(self, p: np.ndarray):
        """Quantile function = percent point function = inverse cdf."""
        lower = self._bc_params["lower"]
        delta = self._qpd_params["delta"]
        kappa = self._qpd_params["kappa"]
        c = self._qpd_params["c"]
        n = self._qpd_params["n"]
        theta = self._qpd_params["theta"]

        phi = self.phi

        in_sinh = np.arcsinh(phi.ppf(p) * delta) + np.arcsinh(n * c * delta)
        in_exp = kappa * np.sinh(in_sinh)
        ppf_arr = lower + theta * np.exp(in_exp)

        return ppf_arr

    def _pdf(self, x: np.ndarray):
        """Probability density function."""
        lower = self._bc_params["lower"]
        delta = self._qpd_params["delta"]
        kappa = self._qpd_params["kappa"]
        c = self._qpd_params["c"]
        n = self._qpd_params["n"]
        theta = self._qpd_params["theta"]

        phi = self.phi

        # we work through the chain rule for the entire nested expression in cdf
        x_ = (x - lower) / theta
        x_der = 1 / theta

        in_arcsinh = np.log(x_) / kappa
        in_arcsinh_der = x_der / (kappa * x_)

        in_sinh = np.arcsinh(in_arcsinh) - np.arcsinh(n * c * delta)
        in_sinh_der = arcsinh_der(in_arcsinh) * in_arcsinh_der

        in_cdf = np.sinh(in_sinh) / delta
        in_cdf_der = np.cosh(in_sinh) * in_sinh_der / delta

        # cdf_arr = phi.cdf(in_cdf)
        cdf_arr_der = phi.pdf(in_cdf) * in_cdf_der

        pdf_arr = cdf_arr_der
        return pdf_arr

    def _cdf(self, x: np.ndarray):
        """Cumulative distribution function."""
        lower = self._bc_params["lower"]
        delta = self._qpd_params["delta"]
        kappa = self._qpd_params["kappa"]
        c = self._qpd_params["c"]
        n = self._qpd_params["n"]
        theta = self._qpd_params["theta"]

        phi = self.phi

        in_arcsinh = np.log((x - lower) / theta) / kappa
        in_sinh = np.arcsinh(in_arcsinh) - np.arcsinh(n * c * delta)
        cdf_arr = phi.cdf(np.sinh(in_sinh) / delta)

        return cdf_arr

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
            "index": pd.RangeIndex(1),
            "columns": pd.Index(["a"]),
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [[-0.3], [-0.2], [-0.1]],
            "qv_median": [[-0.1], [0.0], [0.1]],
            "qv_high": [[0.2], [0.3], [0.4]],
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
    alpha : float or array_like[float]
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float or array_like[float]
        quantile function value of ``alpha``
    qv_median : float or array_like[float]
        quantile function value of quantile 0.5
    qv_high : float or array_like[float]
        quantile function value of quantile ``1 - alpha``
    lower : float or array_like[float]
        lower bound of semi-bounded range.
        This is used when estimating QPD and calculating
        expectation and variance
    upper : float or array_like[float]
        upper bound of semi-bounded range.
        This is used when estimating QPD and calculating
        expectation and variance
    version: str, optional, default="normal"
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
        "authors": ["setoguchi-naoki", "felix-wick", "fkiraly"],
        "maintainers": ["setoguchi-naoki"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf", "pdf"],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
        "broadcast_params": [
            "alpha",
            "qv_low",
            "qv_median",
            "qv_high",
            "lower",
            "upper",
        ],
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

        # precompute parameters for methods
        phi = _resolve_phi(version)
        self.phi = phi

        qpd_params = _prep_qpd_vars(phi=phi, mode="B", **self._bc_params)
        self._qpd_params = qpd_params

    def _ppf(self, p: np.ndarray):
        """Quantile function = percent point function = inverse cdf."""
        lower = self._bc_params["lower"]
        rnge = self._qpd_params["rnge"]
        delta = self._qpd_params["delta"]
        kappa = self._qpd_params["kappa"]
        c = self._qpd_params["c"]
        n = self._qpd_params["n"]
        xi = self._qpd_params["xi"]

        phi = self.phi

        in_cdf = xi + kappa * np.sinh(delta * (phi.ppf(p) + n * c))
        ppf_arr = lower + rnge * phi.cdf(in_cdf)
        return ppf_arr

    def _pdf(self, x: np.ndarray):
        """Probability density function."""
        lower = self._bc_params["lower"]
        rnge = self._qpd_params["rnge"]
        delta = self._qpd_params["delta"]
        kappa = self._qpd_params["kappa"]
        c = self._qpd_params["c"]
        n = self._qpd_params["n"]
        xi = self._qpd_params["xi"]

        phi = self.phi

        # we work through the chain rule for the entire nested expression in cdf
        x_ = (x - lower) / rnge
        x_der = 1 / rnge

        phi_ppf = phi.ppf(x_)
        # derivative of ppf at z is 1 / pdf(ppf(z))
        phi_ppf_der = x_der / phi.pdf(phi.ppf(x_))

        in_arcsinh = (phi_ppf - xi) / kappa
        in_arcsinh_der = phi_ppf_der / kappa

        in_cdf = np.arcsinh(in_arcsinh) / delta - n * c
        in_cdf_der = arcsinh_der(in_arcsinh) * in_arcsinh_der / delta

        # cdf_arr = phi.cdf(in_cdf)
        cdf_arr_der = phi.pdf(in_cdf) * in_cdf_der

        pdf_arr = cdf_arr_der
        return pdf_arr

    def _cdf(self, x: np.ndarray):
        """Cumulative distribution function."""
        lower = self._bc_params["lower"]
        rnge = self._qpd_params["rnge"]
        delta = self._qpd_params["delta"]
        kappa = self._qpd_params["kappa"]
        c = self._qpd_params["c"]
        n = self._qpd_params["n"]
        xi = self._qpd_params["xi"]

        phi = self.phi

        phi_ppf = phi.ppf((x - lower) / rnge)
        in_cdf = np.arcsinh((phi_ppf - xi) / kappa) / delta - n * c
        cdf_arr = phi.cdf(in_cdf)

        return cdf_arr

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
            "index": pd.RangeIndex(1),
            "columns": pd.Index(["a"]),
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [[-0.3], [-0.2], [-0.1]],
            "qv_median": [[-0.1], [0.0], [0.1]],
            "qv_high": [[0.2], [0.3], [0.4]],
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
    lower : float, default = -1e3
        lower bound of probability density function to
        calculate expected value and variance
        expectation and variance
    upper : float, default = 1e3
        upper bound of probability density function to
        calculate expected value and variance
    version: str, optional, default="normal"
        options are ``normal`` (default) or ``logistic``
    dist_shape: float, optional, default=0.0
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
        "authors": ["setoguchi-naoki", "felix-wick", "fkiraly"],
        "maintainers": ["setoguchi-naoki"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf", "pdf"],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
        "broadcast_params": [
            "alpha",
            "qv_low",
            "qv_median",
            "qv_high",
            "lower",
            "upper",
        ],
    }

    def __init__(
        self,
        alpha: float,
        qv_low: float | Sequence,
        qv_median: float | Sequence,
        qv_high: float | Sequence,
        lower: float = -1e3,
        upper: float = 1e3,
        version: str | None = "normal",
        dist_shape: float | None = 0.0,
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
        self.dist_shape = dist_shape
        self.index = index
        self.columns = columns

        super().__init__(index=index, columns=columns)

        # precompute parameters for methods
        phi = _resolve_phi(version)
        self.phi = phi

        qpd_params = _prep_qpd_vars(phi=phi, mode="U", **self._bc_params)
        self._qpd_params = qpd_params

    def _ppf(self, p: np.ndarray):
        """Quantile function = percent point function = inverse cdf."""
        alpha = self._bc_params["alpha"]
        xi = self._qpd_params["xi"]
        gamma = self._qpd_params["gamma"]
        delta = self._qpd_params["delta"]
        kappa = self._qpd_params["kappa"]

        phi = self.phi

        width = phi.ppf(1 - alpha)
        qs = phi.ppf(p) / width

        ppf_arr = xi + kappa * np.sinh((qs - gamma) / delta)
        return ppf_arr

    def _pdf(self, x: np.ndarray):
        """Probability density function."""
        alpha = self._bc_params["alpha"]
        xi = self._qpd_params["xi"]
        gamma = self._qpd_params["gamma"]
        delta = self._qpd_params["delta"]
        kappa = self._qpd_params["kappa"]

        phi = self.phi

        width = phi.ppf(1 - alpha)

        qs = gamma + delta * np.arcsinh((x - xi) / kappa)
        qs_der = delta * arcsinh_der((x - xi) / kappa) / kappa

        # cdf_arr = phi.cdf(qs * width)
        pdf_arr = phi.pdf(qs * width) * qs_der
        return pdf_arr

    def _cdf(self, x: np.ndarray):
        """Cumulative distribution function."""
        alpha = self._bc_params["alpha"]
        xi = self._qpd_params["xi"]
        gamma = self._qpd_params["gamma"]
        delta = self._qpd_params["delta"]
        kappa = self._qpd_params["kappa"]

        phi = self.phi

        width = phi.ppf(1 - alpha)
        qs = gamma + delta * np.arcsinh((x - xi) / kappa)

        cdf_arr = phi.cdf(qs * width)
        return cdf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": -0.3,
            "qv_median": 0.0,
            "qv_high": 0.3,
            "index": pd.RangeIndex(1),
            "columns": pd.Index(["a"]),
        }
        params2 = {
            "alpha": 0.2,
            "version": "normal",
            "qv_low": [[-0.3], [-0.2], [-0.1]],
            "qv_median": [[-0.1], [0.0], [0.1]],
            "qv_high": [[0.2], [0.3], [0.4]],
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a"]),
        }
        return [params1, params2]


def _resolve_phi(phi):
    """Resolve base distribution."""
    if phi == "normal":
        return norm()
    elif phi == "logistic":
        return logistic()
    else:
        return phi


def _prep_qpd_vars(
    alpha,
    qv_low,
    qv_median,
    qv_high,
    lower,
    upper,
    phi,
    mode="B",
    **kwargs,
):
    """Prepare parameters for Johnson Quantile-Parameterized Distributions.

    Parameters
    ----------
    alpha : 2D np.array
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : 2D np.array
        quantile function value of ``alpha``
    qv_median : 2D np.array
        quantile function value of quantile 0.5
    qv_high : 2D np.array
        quantile function value of quantile ``1 - alpha``
    lower : 2D np.array
        lower bound of range.
    upper : 2D np.array
        upper bound of range.
    phi : scipy.stats.rv_continuous
        base distribution
    mode : str
        options are ``B`` (default) or ``S``
        B = bounded mode, S = lower semi-bounded mode
    """
    c = phi.ppf(1 - alpha)

    if mode == "U":
        lower = 0

    qll = qv_low - lower
    qml = qv_median - lower
    qhl = qv_high - lower

    if mode == "B":
        rnge = upper - lower

        def tfun(x):
            return phi.ppf(x / rnge)

    elif mode == "S":
        tfun = np.log
    elif mode == "U":

        def tfun(x):
            return x

    L = tfun(qll)
    H = tfun(qhl)
    B = tfun(qml)
    HL = H - L
    BL = B - L
    HB = H - B
    LH2B = L + H - 2 * B

    HBL = np.where(BL < HB, BL, HB)

    n = np.where(LH2B > 0, 1, -1)
    n = np.where(LH2B == 0, 0, n)

    if mode in ["B", "U"]:
        xi = np.where(LH2B > 0, L, H)
        xi = np.where(LH2B == 0, B, xi)
    if mode == "S":
        theta = np.where(LH2B > 0, qll, qhl)
        theta = np.where(LH2B == 0, qml, theta)
    if mode == "U":
        theta = np.where(LH2B > 0, BL / HL, HB / HL)

    if mode in ["B", "S"]:
        in_arccosh = HL / (2 * HBL)
        delta_unn = np.arccosh(in_arccosh)
        if mode == "S":
            delta_unn = np.sinh(delta_unn)
        delta = delta_unn / c
    elif mode == "U":
        delta = 1.0 / np.arccosh(1 / (2.0 * theta))
        delta = np.where(LH2B == 0, 1, delta)

    if mode == "B":
        kappa = HL / np.sinh(2 * delta * c)
    elif mode == "S":
        kappa = HBL / (delta * c)
    elif mode == "U":
        kappa = HL / np.sinh(2.0 / delta)
        kappa = np.where(LH2B == 0, HB, kappa)

    params = {
        "c": c,
        "L": L,
        "H": H,
        "B": B,
        "n": n,
        "delta": delta,
        "kappa": kappa,
    }

    if mode == "S":
        params["theta"] = theta
    if mode == "B":
        params["rnge"] = rnge
    if mode in ["B", "U"]:
        params["xi"] = xi
    if mode == "U":
        params["gamma"] = -np.sign(LH2B)

    return params


def arcsinh_der(x):
    """Return derivative of arcsinh."""
    return 1 / np.sqrt(1 + x**2)
