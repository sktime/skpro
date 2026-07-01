import numpy as np
from numpy import exp, sinh, arcsinh, arccosh
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import norm, gamma, nbinom, logistic, mstats
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.base import BaseEstimator

from typing import Optional, Union, Tuple


class J_QPD_S:
    """
    Implementation of the semi-bounded mode of Johnson Quantile-Parameterized
    Distributions (J-QPD), see https://repositories.lib.utexas.edu/bitstream/handle/2152/63037/HADLOCK-DISSERTATION-2017.pdf
    (Due to the Python keyword, the parameter lambda from this reference is named kappa below.).
    A distribution is parameterized by a symmetric-percentile triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : np.ndarray
        quantile function values of ``alpha``
    qv_median : np.ndarray
        quantile function values of quantile 0.5
    qv_high : np.ndarray
        quantile function values of quantile ``1 - alpha``
    l : float
        lower bound of semi-bounded range (default is 0)
    version: str
        options are ``normal`` (default) or ``logistic``
    """

    def __init__(
        self,
        alpha: float,
        qv_low: Union[float, np.ndarray],
        qv_median: Union[float, np.ndarray],
        qv_high: Union[float, np.ndarray],
        l: Optional[float] = 0,
        version: Optional[str] = "normal",
    ):
        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        qv_low = np.asarray(qv_low)
        qv_median = np.asarray(qv_median)
        qv_high = np.asarray(qv_high)

        if (qv_low > qv_median).any() or (qv_high < qv_median).any():
            raise ValueError("The SPT values need to be monotonically increasing.")

        self.l = l

        self.c = self.phi.ppf(1 - alpha)

        self.L = np.log(qv_low - l)
        self.H = np.log(qv_high - l)
        self.B = np.log(qv_median - l)

        self.n = np.where(self.L + self.H - 2 * self.B > 0, 1, -1)
        self.theta = np.where(self.L + self.H - 2 * self.B > 0, qv_low - l, qv_high - l)

        self.n = np.where(self.L + self.H - 2 * self.B == 0, 0, self.n)
        self.theta = np.where(self.L + self.H - 2 * self.B == 0, qv_median - l, self.theta)

        self.delta = (
            1.0
            / self.c
            * np.sinh(
                np.arccosh(
                    (self.H - self.L)
                    / (2 * np.where((self.B - self.L) < (self.H - self.B), (self.B - self.L), (self.H - self.B)))
                )
            )
        )

        self.kappa = (
            1.0
            / (self.delta * self.c)
            * np.where((self.H - self.B) < (self.B - self.L), (self.H - self.B), (self.B - self.L))
        )

    def ppf(self, x: Union[float, np.ndarray], inner=False) -> Union[float, np.ndarray]:
        """
        Percent point function (inverse of cdf).

        Parameters
        ----------
        x : np.ndarray
            quantiles to be calculated
        inner : bool
            flag to choose between inner (True) or outer (False) vector
            multiplication of QPD distributions for a set of samples and
            quantiles to be calculated, default (outer)

        Returns
        -------
        np.ndarray
            values according to the quantiles given
        """
        if inner:
            ppf_value = self.l + self.theta * exp(
                self.kappa
                * np.sinh(np.arcsinh(self.phi.ppf(x) * self.delta) + np.arcsinh(self.n * self.c * self.delta))
            )
        else:
            ppf_value = self.l + self.theta * exp(
                self.kappa
                * np.sinh(np.arcsinh(np.outer(self.phi.ppf(x), self.delta)) + np.arcsinh(self.n * self.c * self.delta))
            )

        if ppf_value.ndim == 0:
            ppf_value = ppf_value.item()
        ppf_value = np.squeeze(ppf_value)

        return ppf_value

    def cdf(self, x: Union[float, np.ndarray], inner=False) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function.

        Parameters
        ----------
        x : np.ndarray
            values to be calculated
        inner : bool
            flag to choose between inner (True) or outer (False) vector
            multiplication of QPD distributions for a set of samples and
            values to be calculated, default (outer)

        Returns
        -------
        np.ndarray
            quantiles
        """
        if inner:
            cdf_value = self.phi.cdf(
                1.0
                / self.delta
                * np.sinh(
                    np.arcsinh(1.0 / self.kappa * np.log((x - self.l) / self.theta))
                    - np.arcsinh(self.n * self.c * self.delta)
                )
            )
        else:
            cdf_value = self.phi.cdf(
                1.0
                / self.delta
                * np.sinh(
                    np.arcsinh(1.0 / self.kappa * np.log(np.outer((x - self.l), 1.0 / self.theta)))
                    - np.arcsinh(self.n * self.c * self.delta)
                )
            )

        if cdf_value.ndim == 0:
            cdf_value = cdf_value.item()
        cdf_value = np.squeeze(cdf_value)

        return cdf_value


class J_QPD_B:
    """
    Implementation of the bounded mode of Johnson Quantile-Parameterized
    Distributions (J-QPD), see https://repositories.lib.utexas.edu/bitstream/handle/2152/63037/HADLOCK-DISSERTATION-2017.pdf.
    (Due to the Python keyword, the parameter lambda from this reference is named kappa below.)
    A distribution is parameterized by a symmetric-percentile triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : np.ndarray
        quantile function values of ``alpha``
    qv_median : np.ndarray
        quantile function values of quantile 0.5
    qv_high : np.ndarray
        quantile function values of quantile ``1 - alpha``
    l : float
        lower bound of supported range
    u : float
        upper bound of supported range
    version: str
        options are ``normal`` (default) or ``logistic``
    """

    def __init__(
        self,
        alpha: float,
        qv_low: Union[float, np.ndarray],
        qv_median: Union[float, np.ndarray],
        qv_high: Union[float, np.ndarray],
        l: float,
        u: float,
        version: Optional[str] = "normal",
    ):
        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        qv_low = np.asarray(qv_low)
        qv_median = np.asarray(qv_median)
        qv_high = np.asarray(qv_high)

        if (qv_low > qv_median).any() or (qv_high < qv_median).any():
            raise ValueError("The SPT values need to be monotonically increasing.")

        self.l = l
        self.u = u

        self.c = self.phi.ppf(1 - alpha)

        self.L = self.phi.ppf((qv_low - l) / (u - l))
        self.H = self.phi.ppf((qv_high - l) / (u - l))
        self.B = self.phi.ppf((qv_median - l) / (u - l))

        self.n = np.where(self.L + self.H - 2 * self.B > 0, 1, -1)
        self.xi = np.where(self.L + self.H - 2 * self.B > 0, self.L, self.H)

        self.n = np.where(self.L + self.H - 2 * self.B == 0, 0, self.n)
        self.xi = np.where(self.L + self.H - 2 * self.B == 0, self.B, self.xi)

        self.delta = (
            1.0
            / self.c
            * np.arccosh(
                (self.H - self.L)
                / (2 * np.where((self.B - self.L) < (self.H - self.B), self.B - self.L, self.H - self.B))
            )
        )

        self.kappa = (self.H - self.L) / np.sinh(2 * self.delta * self.c)

    def ppf(self, x: Union[float, np.ndarray], inner=False) -> Union[float, np.ndarray]:
        if inner:
            ppf_value = self.l + (self.u - self.l) * self.phi.cdf(
                self.xi + self.kappa * np.sinh(self.delta * (self.phi.ppf(x) + self.n * self.c))
            )
        else:
            if np.isscalar(x):
                x = np.asarray([x])
            ppf_value = self.l + (self.u - self.l) * self.phi.cdf(
                self.xi + self.kappa * np.sinh(self.delta * (self.phi.ppf(x[:, np.newaxis]) + self.n * self.c))
            )

        if ppf_value.ndim == 0:
            ppf_value = ppf_value.item()
        ppf_value = np.squeeze(ppf_value)

        return ppf_value

    def cdf(self, x: Union[float, np.ndarray], inner=False) -> Union[float, np.ndarray]:
        if inner:
            cdf_value = self.phi.cdf(
                1.0
                / self.delta
                * np.arcsinh(1.0 / self.kappa * (self.phi.ppf((x - self.l) / (self.u - self.l)) - self.xi))
                - self.n * self.c
            )
        else:
            if np.isscalar(x):
                x = np.asarray([x])
            cdf_value = self.phi.cdf(
                1.0
                / self.delta
                * np.arcsinh(
                    1.0 / self.kappa * (self.phi.ppf((x[:, np.newaxis] - self.l) / (self.u - self.l)) - self.xi)
                )
                - self.n * self.c
            )

        if cdf_value.ndim == 0:
            cdf_value = cdf_value.item()
        cdf_value = np.squeeze(cdf_value)

        return cdf_value


class SinhLogistic:
    """
    sinh/arcsinh-modified logistic distribution for smooth interpolation
    between logistic and t2 distribution.

    Parameters
    ----------
    shape: float
        parameter modifying the logistic base distribution via
        sinh/arcsinh-scaling
    """

    def __init__(self, shape: float):
        self.shape = shape

    def ppf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # ppf of natural logistic distribution
        xlog = 0.25 * np.log(x / (1.0 - x))

        # sinh or arcsinh scaling
        if self.shape > 0:
            x = np.arcsinh(self.shape * xlog) / self.shape
        elif self.shape < 0:
            x = np.sinh(self.shape * xlog) / self.shape
        else:
            x = xlog
        return x

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # sinh or arcsinh scaling
        if self.shape > 0:
            xlog = np.sinh(self.shape * x) / self.shape
        elif self.shape < 0:
            xlog = np.arcsinh(self.shape * x) / self.shape
        else:
            xlog = x

        # natural logistic cdf
        return 1.0 / (1 + np.exp(-4 * xlog))

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.shape > 0:
            return np.cosh(self.shape * x) / (np.cosh(2.0 / self.shape * np.sinh(self.shape * x))) ** 2
        elif self.shape < 0:
            return (
                1.0 / np.sqrt((self.shape * x) ** 2 + 1) / (np.cosh(2.0 / self.shape * np.arcsinh(self.shape * x))) ** 2
            )
        else:
            return 1.0 / (np.cosh(2 * x)) ** 2


class BaseDist:
    """
    Scaling of base ppfs such that ppf(1 - alpha) = 1. A detailed description
    can be found in the presentation JQPDregression.pdf in the docs folder of
    this repository.
    """

    def __init__(self, dist, alpha: float):
        self.dist = dist
        self.alpha = alpha
        self.width = self.dist.ppf(1 - self.alpha)

    def ppf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.dist.ppf(x) / self.width

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.dist.cdf(x * self.width)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.dist.pdf(x * self.width) * self.width


def unconstrained_calc(L: float, B: float, H: float) -> Tuple[float, float, float, float]:
    gamma = -np.sign(L + H - 2 * B)

    if gamma == 0:
        xi = B
        kappa = H - B
        delta = 1
    else:
        if gamma < 0:
            xi = L
            theta = (B - L) / (H - L)
        else:
            xi = H
            theta = (H - B) / (H - L)
        delta = 1.0 / arccosh(1 / (2.0 * theta))
        kappa = (H - L) / sinh(2.0 / delta)

    return gamma, xi, kappa, delta


class J_QPD_extended_U:
    """
    Unbounded version of J-QPDs (see bounded and semi-bounded versions above),
    including an additional shape parameter to enable flexible tail behavior.
    Again, a distribution is parameterized by a symmetric-percentile triplet
    (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float
        quantile function value of ``alpha``
    qv_median : float
        quantile function value of quantile 0.5
    qv_high : float
        quantile function value of quantile ``1 - alpha``
    version: str
        options are ``normal`` (sinhlogistic), ``normal`, or ``logistic``
    shape: float
        parameter modifying the logistic base distribution via
        sinh/arcsinh-scaling (only active in sinhlogistic version)
    """

    def __init__(
        self,
        alpha: float,
        qv_low: float,
        qv_median: float,
        qv_high: float,
        version: Optional[str] = "sinhlogistic",
        shape: Optional[float] = 0,
    ):
        self.shape = shape
        self.alpha = alpha

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        elif version == "sinhlogistic":
            self.phi = SinhLogistic(shape=self.shape)
        else:
            raise Exception("Invalid version.")

        if (qv_low > qv_median) or (qv_high < qv_median):
            raise ValueError("The SPT values need to be monotonically increasing.")

        # identity transformation
        self.gamma, self.xi, self.kappa, self.delta = unconstrained_calc(qv_low, qv_median, qv_high)

    def ppf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        basequantiles = BaseDist(self.phi, self.alpha).ppf(x)

        # internal unconstrained quantiles from Johnson transform
        # back transformatiaon into physical space (identity here)
        return self.xi + self.kappa * sinh((basequantiles - self.gamma) / self.delta)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # transform from physical to internal space (identity here)
        # internal unconstrained quantiles from Johnson transform
        basequantiles = self.gamma + self.delta * arcsinh((x - self.xi) / self.kappa)

        # cdf of base distribution
        return BaseDist(self.phi, self.alpha).cdf(basequantiles)


class J_QPD_extended_S:
    """
    Semi-bounded version of J-QPDs, extended by a shape parameter to enable
    flexible tail behavior. A distribution is parameterized by a
    symmetric-percentile triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float
        quantile function value of ``alpha``
    qv_median : float
        quantile function value of quantile 0.5
    qv_high : float
        quantile function value of quantile ``1 - alpha``
    l : float
        lower bound of semi-bounded range (default is 0)
    version: str
        options are ``normal`` (sinhlogistic), ``normal`, or ``logistic``
    shape: float
        parameter modifying the logistic base distribution via
        sinh/arcsinh-scaling (only active in sinhlogistic version)
    """

    def __init__(
        self,
        alpha: float,
        qv_low: float,
        qv_median: float,
        qv_high: float,
        l: Optional[float] = 0,
        version: Optional[str] = "sinhlogistic",
        shape: Optional[float] = 0,
    ):
        self.shape = shape
        self.alpha = alpha

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        elif version == "sinhlogistic":
            self.phi = SinhLogistic(shape=self.shape)
        else:
            raise Exception("Invalid version.")

        if (qv_low > qv_median) or (qv_high < qv_median):
            raise ValueError("The SPT values need to be monotonically increasing.")

        self.l = l

        # transform input quantiles from semi-(lower-)bounded physical space to unconstrained internal space
        self.L = transform_from_semibound_lower(qv_low, self.l)
        self.H = transform_from_semibound_lower(qv_high, self.l)
        self.B = transform_from_semibound_lower(qv_median, self.l)

        # now handle like unconstrained
        self.gamma, self.xi, self.kappa, self.delta = unconstrained_calc(self.L, self.B, self.H)

    def ppf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        basequantiles = BaseDist(self.phi, self.alpha).ppf(x)

        # internal unconstrained quantiles from Johnson transform
        z = self.xi + self.kappa * sinh((basequantiles - self.gamma) / self.delta)

        # back transformation into physical space
        return back_transform_in_semibound_lower(z, self.l)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # transform from physical to internal space
        z = transform_from_semibound_lower(x, self.l)

        # internal unconstrained quantiles from Johnson transform
        basequantiles = self.gamma + self.delta * arcsinh((z - self.xi) / self.kappa)

        return BaseDist(self.phi, self.alpha).cdf(basequantiles)


class J_QPD_extended_B:
    """
    Bounded version of J-QPDs, extended by a shape parameter to enable flexible
    tail behavior. A distribution is parameterized by a symmetric-percentile
    triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float
        quantile function value of ``alpha``
    qv_median : float
        quantile function value of quantile 0.5
    qv_high : float
        quantile function value of quantile ``1 - alpha``
    l : float
        lower bound of supported range
    u : float
        upper bound of supported range
    version: str
        options are ``normal`` (sinhlogistic), ``normal`, or ``logistic``
    shape: float
        parameter modifying the logistic base distribution via
        sinh/arcsinh-scaling (only active in sinhlogistic version)
    """

    def __init__(
        self,
        alpha: float,
        qv_low: float,
        qv_median: float,
        qv_high: float,
        l: Optional[float] = 0,
        u: Optional[float] = 1,
        version: Optional[str] = "sinhlogistic",
        shape: Optional[float] = 0,
    ):
        self.shape = shape
        self.alpha = alpha

        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        elif version == "sinhlogistic":
            self.phi = SinhLogistic(shape=self.shape)
        else:
            raise Exception("Invalid version.")

        if (qv_low > qv_median) or (qv_high < qv_median):
            raise ValueError("The SPT values need to be monotonically increasing.")

        self.l = l
        self.u = u

        # transform input quantiles from bounded physical space to unconstrained internal space
        self.L = transform_from_bounds(qv_low, self.l, self.u)
        self.H = transform_from_bounds(qv_high, self.l, self.u)
        self.B = transform_from_bounds(qv_median, self.l, self.u)

        # now handle like unconstrained
        self.gamma, self.xi, self.kappa, self.delta = unconstrained_calc(self.L, self.B, self.H)

    def ppf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        basequantiles = BaseDist(self.phi, self.alpha).ppf(x)

        # internal unconstrained quantiles from Johnson transform
        z = self.xi + self.kappa * sinh((basequantiles - self.gamma) / self.delta)

        # back transformation into [l, u]
        return back_transform_in_bounds(z, self.l, self.u)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # transform from bounded physical space to unconstrained internal space
        z = transform_from_bounds(x, self.l, self.u)

        # internal unconstrained quantiles from Johnson transform
        basequantiles = self.gamma + self.delta * arcsinh((z - self.xi) / self.kappa)

        # cdf of base distribution
        p = BaseDist(self.phi, self.alpha).cdf(basequantiles)

        return p


def transform_from_bounds(x: np.ndarray, l: float, u: float) -> np.ndarray:
    # transform from bounded physical space to [0, 1]
    z = (x - l) / (u - l)

    # transfer to unconstrained internal space by logit
    z = z / (1 - z)
    z = np.where(z == 0, 1e-12, z)
    return np.log(z)


def back_transform_in_bounds(z: np.ndarray, l: float, u: float) -> np.ndarray:
    # back transformation into [0, 1]
    x = 1.0 / (1.0 + np.exp(-z))

    # back transformation into [l, u]
    return l + (u - l) * x


def transform_from_semibound_lower(x: np.ndarray, l: float) -> np.ndarray:
    z = x - l
    z = np.where(z == 0, 1e-12, z)

    # transfer to unconstrained internal space
    return np.log(z)


def back_transform_in_semibound_lower(z: np.ndarray, l: float) -> np.ndarray:
    return l + np.exp(z)


def transform_from_semibound_upper(x: np.ndarray, u: float) -> np.ndarray:
    z = u - x
    z = np.where(z == 0, 1e-12, z)

    # transfer to unconstrained internal space
    return np.log(z)


def back_transform_in_semibound_upper(z: np.ndarray, u: float) -> np.ndarray:
    return u - np.exp(z)


def fit_sinhlogistic_shape(alpha: float, l: float, u: float, bound: str, y: np.ndarray) -> float:
    """
    Fit of the shape parameter of our extended versions of the J-QPD mechanism,
    which modifies the logistic base distribution by sinh/arcsinh-scaling, on
    the empirical quantile function of the observations.
    """

    if bound not in ["S", "B", "U"]:
        raise Exception("Invalid version.")

    qlowincl = np.quantile(y, alpha)
    qmedincl = np.quantile(y, 0.5)
    qhighincl = np.quantile(y, 1 - alpha)

    if bound == "S":
        jqpd_inclusive = J_QPD_extended_S(alpha, qlowincl, qmedincl, qhighincl, l)
    elif bound == "B":
        jqpd_inclusive = J_QPD_extended_B(alpha, qlowincl, qmedincl, qhighincl, l, u)
    else:
        jqpd_inclusive = J_QPD_extended_U(alpha, qlowincl, qmedincl, qhighincl)

    def fit_shape(shape, p, q):
        jqpd_inclusive.shape = shape
        jqpd_inclusive.phi = SinhLogistic(shape=jqpd_inclusive.shape)

        q_fit = jqpd_inclusive.ppf(p)

        emd = (np.abs(q - q_fit)).mean()
        return emd

    stepsize = 0.001
    p = np.arange(stepsize / 2.0, 1 - stepsize / 2.0, stepsize)
    q = mstats.mquantiles(y, p)
    param = minimize_scalar(fit_shape, args=(p, q))
    return param.x


class QPD_RegressorChain(BaseEstimator):
    """
    Constrained conditional density estimation by means of quantile regression
    and quantile-parameterized distributions (QPD).

    The training chain consists of several steps:

    First, the median is trained and predicted in-sample by means of an
    arbitrary method (e.g., an ML regressor using a pinball loss, like Cyclic
    Boosting). External (and also internal ones, see below) constraints on the
    target range can be taken into account by non-linear transformations,
    exploiting the bijectivity of quantile transformations (Quantile of
    transformed variable equals transformation of quantile of original
    variable.).

    Next, the lower quantile of the QPD symmetric-percentile triplet (SPT) is
    trained and predicted in-sample by means of an arbitrary method (again
    taking into account external constraints on the target range), which is
    independent, and therefore can be completely different, to the median model
    above. Due to the internal constraint to train only on samples with target
    value lower than the in-sample-predicted median, the quantile that must
    actually be used in the quantile regression method is ``2 * alpha``.

    Next, the upper quantile of the QPD SPT triplet is trained by means of an
    arbitrary method (again taking into account external constraints on the
    target range), which is independent, and therefore can be completely
    different, to the median and lower quantile models above. Due to the
    internal constraint to train only on samples with target value higher than
    the in-sample-predicted median, the quantile that must actually be used in
    the quantile regression method is ``2 * (1 - alpha) - 1``.

    The prediction chain is then executed accordingly, and finally, a QPD is
    calculated for each sample by means of the predicted SPT. For this,
    extended versions of the J-QPD mechanism are used, which include a shape
    parameter modifying the logistic base distribution by sinh/arcsinh-scaling.
    This shape parameter is independently fitted on the empirical quantile
    function of the training targets.

    Parameters
    ----------
    est_median : BaseEstimator
        estimator to be used predict the median
    est_lowq : BaseEstimator
        estimator to be used predict the lower quantile ``alpha`` (must be
        trained with quantile ``2 * alpha``, e.g., median for ``alpha = 0.25``)
    est_highq : BaseEstimator
        estimator to be used predict the upper quantile ``1 - alpha`` (must be
        trained with quantile ``2 * (1 - alpha) - 1``, e.g., median for
        ``alpha = 0.25``)
    bound: str
        Different modes defined by supported target range, options are ``S``
        (semi-bound) and ``B`` (bound).
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    l : float
        lower bound of supported range
    u : float
        upper bound of supported range (only active for bound mode)
    """

    def __init__(
        self,
        est_median: BaseEstimator,
        est_lowq: BaseEstimator,
        est_highq: BaseEstimator,
        bound: str,
        alpha: Optional[float] = 0.25,
        l: Optional[float] = 0,
        u: Optional[float] = 1,
    ):
        self.est_median = est_median
        self.est_lowq = est_lowq
        self.est_highq = est_highq  # trained with quantile 2 * (1 - alpha) - 1 (e.g., median for 0.75)

        self.alpha = alpha

        self.l = l
        self.u = u

        self.bound = bound
        if self.bound not in ["S", "B"]:
            raise Exception("Invalid version.")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> BaseEstimator:
        # median model
        if self.bound == "S":
            y_trans = transform_from_semibound_lower(y, self.l)
        elif self.bound == "B":
            y_trans = transform_from_bounds(y, self.l, self.u)
        self.est_median.fit(X, y_trans)
        z = self.est_median.predict(X)
        if self.bound == "S":
            pred_median = back_transform_in_semibound_lower(z, self.l)
        elif self.bound == "B":
            pred_median = back_transform_in_bounds(z, self.l, self.u)

        # lower quantile model
        y_trans = y / pred_median
        mask = y_trans < 1
        y_trans = y_trans[mask]
        y_trans = transform_from_bounds(y_trans, self.l / pred_median[mask], 1)
        self.est_lowq.fit(X[mask], y_trans)
        z = self.est_lowq.predict(X)
        pred_lowq = back_transform_in_bounds(z, self.l, pred_median)

        # upper quantile model
        y_trans = (y - pred_median) / (pred_median - pred_lowq)
        mask = y_trans > 0
        y_trans = y_trans[mask]
        if self.bound == "S":
            y_trans = transform_from_semibound_lower(y_trans, 0)
        elif self.bound == "B":
            y_trans = transform_from_bounds(y_trans, 0, (self.u - pred_median) / (pred_median - pred_lowq))
        self.est_highq.fit(X[mask], y_trans)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> tuple:
        if self.bound == "S":
            pred_median = back_transform_in_semibound_lower(self.est_median.predict(X), self.l)
            pred_lowq = back_transform_in_bounds(self.est_lowq.predict(X), self.l, pred_median)
            pred_highq = back_transform_in_semibound_lower(self.est_highq.predict(X), pred_median)
        elif self.bound == "B":
            pred_median = back_transform_in_bounds(self.est_median.predict(X), self.l, self.u)
            pred_lowq = back_transform_in_bounds(self.est_lowq.predict(X), self.l, pred_median)
            pred_highq = back_transform_in_bounds(self.est_highq.predict(X), pred_median, self.u)

        if self.bound == "S":
            qpd = J_QPD_S(self.alpha, pred_lowq, pred_median, pred_highq, self.l)
        elif self.bound == "B":
            qpd = J_QPD_B(self.alpha, pred_lowq, pred_median, pred_highq, self.l, self.u)

        return pred_lowq, pred_median, pred_highq, qpd


def quantile_fit_gaussian(quantiles: np.ndarray, quantile_values: np.ndarray, mode: Optional[str] = "ppf") -> callable:
    """
    Interpolation of a quantile function (with quantiles estimated, e.g., by
    means of quantile regression) according to a Gaussian distribution as
    assumed PDF.

    Parameters
    ----------
    quantiles : np.ndarray
        quantiles (x values of quantile function)
    quantile_values : np.ndarray
        quantile values (y values of quantile function)
    mode : str
        decides about kind of returned callable, possible values are:

            - ``ppf``: quantile function (default)
            - ``dist``: fitted Gaussian function (scipy function)
            - ``cdf``: CDF function

    Returns
    -------
    callable
        fitted Gaussian function (see mode)
    """

    def f(x, mu, sigma):
        return norm(loc=mu, scale=sigma).ppf(x)

    mu, sigma = curve_fit(f, quantiles, quantile_values)[0]
    if mode == "ppf":
        return norm(mu, sigma).ppf
    elif mode == "dist":
        return norm(mu, sigma)
    elif mode == "cdf":
        return norm(mu, sigma).cdf
    else:
        raise Exception("Invalid mode.")


def quantile_fit_gamma(quantiles: np.ndarray, quantile_values: np.ndarray, mode: Optional[str] = "ppf") -> callable:
    """
    Interpolation of a quantile function (with quantiles estimated, e.g., by
    means of quantile regression) according to a Gamma distribution as assumed
    PDF (i.e., continuous, non-negative target values).

    Parameters
    ----------
    quantiles : np.ndarray
        quantiles (x values of quantile function)
    quantile_values : np.ndarray
        quantile values (y values of quantile function)
    mode : str
        decides about kind of returned callable, possible values are:

            - ``ppf``: quantile function (default)
            - ``dist``: fitted Gamma function (scipy function)
            - ``cdf``: CDF function

    Returns
    -------
    callable
        fitted Gamma function (see mode)
    """

    def f(x, alpha, beta):
        return gamma(alpha, scale=1 / beta).ppf(x)

    alpha, beta = curve_fit(f, quantiles, quantile_values, p0=[2.0, 0.9])[0]
    if mode == "ppf":
        return gamma(alpha, scale=1 / beta).ppf
    elif mode == "dist":
        return gamma(alpha, scale=1 / beta)
    elif mode == "cdf":
        return gamma(alpha, scale=1 / beta).cdf
    else:
        raise Exception("Invalid mode.")


def _nbinom_cdf_mu_var(x: float, mu: float, var: float) -> callable:
    """
    Calculation of negative binomial parameters n and p from given mean and
    variance, and subsequent call of its cumulative distribution function.

    Parameters
    ----------
    x : float
        value of random variable following negative binomial distribution
    mu : float
        mean of negative binomial distribution
    var : float
        variance of negative binomial distribution

    Returns
    -------
    callable
        negative binomial cumulative distribution function
    """
    n = mu * mu / (var - mu)
    p = mu / var
    return nbinom(n, p).cdf(x)


def quantile_fit_nbinom(quantiles: np.ndarray, quantile_values: np.ndarray, mode: Optional[str] = "ppf") -> callable:
    """
    Interpolation of a quantile function (with quantiles estimated, e.g., by
    means of quantile regression) according to a negative binomial distribution
    as assumed PDF (i.e., discrete, non-negative target values).

    Parameters
    ----------
    quantiles : np.ndarray
        quantiles (x values of quantile function)
    quantile_values : np.ndarray
        quantile values (y values of quantile function)
    mode : str
        decides about kind of returned callable, possible values are:

            - ``ppf``: quantile function (default)
            - ``dist``: fitted negative binomial function (scipy function)
            - ``cdf``: CDF function

    Returns
    -------
    callable
        fitted negative binomial function (see mode)
    """
    mu, var = curve_fit(_nbinom_cdf_mu_var, quantile_values, quantiles, p0=[2.2, 2.4])[0]
    n = mu * mu / (var - mu)
    p = mu / var
    if mode == "ppf":
        return nbinom(n, p).ppf
    elif mode == "dist":
        return nbinom(n, p)
    elif mode == "cdf":
        return nbinom(n, p).cdf
    else:
        raise Exception("Invalid mode.")


def quantile_fit_spline(quantiles: np.ndarray, quantile_values: np.ndarray) -> callable:
    """
    Interpolation of a quantile function (with quantiles estimated, e.g., by
    means of quantile regression) according to a smoothing spline (i.e.,
    arbitrary target distribution).

    Parameters
    ----------
    quantiles : np.ndarray
        quantiles (x values of quantile function)
    quantile_values : np.ndarray
        quantile values (y values of quantile function)

    Returns
    -------
    callable
        spline fitted to quantile function
    """
    spl = InterpolatedUnivariateSpline(quantiles, quantile_values, k=3, bbox=[0, 1], ext=3)
    return spl
