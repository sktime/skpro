"""
Cyclic Boosting Negative Binomial c regressor. var = mu + c * mu * mu
"""

import logging
from math import lgamma

import numba as nb
import numpy as np
import sklearn.base

from skpro.libs.cyclic_boosting.base import CyclicBoostingBase
from skpro.libs.cyclic_boosting.learning_rate import constant_learn_rate_one
from skpro.libs.cyclic_boosting.link import LogitLinkMixin
from skpro.libs.cyclic_boosting.utils import get_X_column

_logger = logging.getLogger(__name__)


def _try_compile_parallel_func(**targetoptions):
    """
    Decorator that tries to compile a wrapped function in numba parallel mode. If that
    fails it will compile the function in non-parallel mode.
    """

    def wrapper(f):
        try:
            wrapper = nb.jit(**targetoptions, parallel=True)
            func = wrapper(f)

        except:  # noqa
            _logger.warning(
                f"Could not compile function {f} in parallel mode, falling back to"
                f"non-parallel mode."
            )
            wrapper = nb.jit(**targetoptions, parallel=False)
            func = wrapper(f)

        return func

    return wrapper


class CBNBinomC(CyclicBoostingBase, sklearn.base.RegressorMixin, LogitLinkMixin):
    """Maximum Likelihood estimation of the c parameter of the Negative Binomial
    Negative Binomial Variance: mu + c * mu**2
    Estimator predicts c with the constraint: c in (0,1)
    Follows https://en.wikipedia.org/wiki/Negative_binomial_distribution notation with c = 1/r

    Parameters
    ----------

    mean_prediction_column: string or None
        Column for the mean of the Negative Binomial

    gamma: float
        Lasso term, zero for non-penalized fit. The larger the value the harder the regularization.

    bayes: bool
        use expectation of the posterior instead of maximum likelihood in each cyclic boosting step

    The rest of the parameters are documented in CyclicBoostingBase.
    """

    def __init__(
        self,
        mean_prediction_column,
        feature_groups=None,
        feature_properties=None,
        weight_column=None,
        prior_prediction_column=None,
        minimal_loss_change=1e-4,
        minimal_factor_change=1e-4,
        maximal_iterations=10,
        observers=None,
        smoother_choice=None,
        output_column=None,
        learn_rate=constant_learn_rate_one,
        gamma=0.0,
        bayes=False,
        n_steps=15,
    ):
        CyclicBoostingBase.__init__(
            self,
            feature_groups=feature_groups,
            feature_properties=feature_properties,
            weight_column=weight_column,
            prior_prediction_column=prior_prediction_column,
            minimal_loss_change=minimal_loss_change,
            minimal_factor_change=minimal_factor_change,
            maximal_iterations=maximal_iterations,
            observers=observers,
            smoother_choice=smoother_choice,
            output_column=output_column,
            learn_rate=learn_rate,
        )
        self.mean_prediction_column = mean_prediction_column
        self.gamma = gamma
        self.bayes = bayes
        self.n_steps = n_steps

    def _check_y(self, y):
        """Check that y has no negative values."""
        if not (y >= 0.0).all():
            raise ValueError(
                "The target y must be positive semi-definite "
                "and not NAN. y[~(y>=0)] = {}".format(y[~(y >= 0)])
            )

    def precalc_parameters(self, feature, y, pred):
        return None

    def calc_parameters(self, feature, y, pred, prefit_data):
        c_link = pred.predict_link()
        binnumbers = feature.lex_binned_data
        minlength = feature.n_bins
        new_c_link = get_new_c_link_for_iteration(self.iteration_ + 1, self.n_steps)
        # TODO: use weights
        c_link_estimate = calc_parameters_nbinom_c(
            y,
            self.mu,
            c_link,
            binnumbers,
            minlength,
            self.gamma,
            int(self.bayes),
            new_c_link,
        )

        bincounts = np.bincount(binnumbers, minlength=minlength)
        bincounts[bincounts < 1] = 1.0
        return c_link_estimate, 1.0 / np.sqrt(bincounts)

    def loss(self, c, y, weights):
        # TODO: use weights
        return loss_nbinom_c(y.astype(np.float64), self.mu, c, self.gamma)

    def fit(self, X, y=None):
        self.mu = X[self.mean_prediction_column].values
        _ = self._fit_predict(X, y)
        del self.mu
        return self

    def _get_prior_predictions(self, X):
        if self.prior_prediction_column is None:
            prior_prediction_link = np.repeat(self.neutral_factor_link, X.shape[0])
        else:
            prior_pred = get_X_column(X, self.prior_prediction_column)
            np.clip(prior_pred, 0, 1, prior_pred)
            prior_prediction_link = self.link_func(prior_pred)
            finite = np.isfinite(prior_prediction_link)
            prior_prediction_link[~finite] = self.neutral_factor_link

        return prior_prediction_link

    def _init_global_scale(self, X, y):
        if self.weights is None:
            raise RuntimeError("The weights have to be initialized.")
        self.global_scale_link_ = self.neutral_factor_link
        self.prior_pred_link_offset_ = self.neutral_factor_link


def get_new_c_link_for_iteration(iteration, n_steps):
    new_c_link = np.r_[
        -np.logspace(-3, 1.0 / iteration, n_steps // iteration)[::-1],
        0,
        np.logspace(-3, 1.0 / iteration, n_steps // iteration),
    ]
    return np.ascontiguousarray(new_c_link)


@nb.njit()
def nbinom_log_pmf(x: nb.float64, n: nb.float64, p: nb.float64) -> nb.float64:
    """
    Negative binomial log PMF.
    """
    coeff = lgamma(n + x) - lgamma(x + 1) - lgamma(n)
    return coeff + n * np.log(p) + x * np.log(1 - p)


@_try_compile_parallel_func(
    nogil=True,
    nopython=True,
)
def loss_nbinom_c(
    y: nb.float64[:], mu: nb.float64[:], c: nb.float64[:], gamma: nb.float64
) -> nb.float64:
    n_samples = len(y)

    p = np.minimum(1.0 / (1 + c * mu), 1.0 - 1e-8)
    n = mu * p / (1 - p)

    loss = np.zeros(n_samples)
    for i in nb.prange(n_samples):
        loss[i] = -nbinom_log_pmf(y[i], n[i], p[i])

    loss[~np.isfinite(loss)] = 400
    loss += gamma * np.fabs(c)

    return np.mean(loss)


@nb.njit()
def binned_loss_nbinom_c(
    y: nb.float64[:],
    mu: nb.float64[:],
    c_link: nb.float64[:],
    binnumbers: nb.int64[:],
    minlength: nb.int64,
    gamma: nb.float64,
    new_c_link: nb.float64,
) -> nb.float64[:]:
    n_samples = len(y)

    c = 1.0 / (1.0 + np.exp(-(new_c_link + c_link)))
    p = np.minimum(1.0 / (1.0 + c * mu), 1.0 - 1e-8)
    n = mu * p / (1 - p)

    loss = np.zeros(minlength)
    for i in range(n_samples):
        ibin = binnumbers[i]
        loss_i = -nbinom_log_pmf(y[i], n[i], p[i]) + gamma * np.fabs(c[i])
        if np.isfinite(loss_i):
            loss[ibin] += loss_i
        else:
            loss[ibin] += 400 + gamma * np.fabs(c[i])

    return loss


@_try_compile_parallel_func(
    nogil=True,
    nopython=True,
)
def compute_2d_loss(
    y: nb.float64[:],
    mu: nb.float64[:],
    c_link: nb.float64[:],
    binnumbers: nb.int64[:],
    minlength: nb.int64,
    gamma: nb.float64,
    new_c_link: nb.float64[:],
) -> nb.float64[:, :]:
    n_new_c = len(new_c_link)
    loss = np.empty((n_new_c, minlength), dtype=np.float64)

    for i in nb.prange(n_new_c):
        loss[i] = binned_loss_nbinom_c(
            y.astype(np.float64),
            mu,
            c_link,
            binnumbers,
            minlength,
            gamma,
            new_c_link[i],
        )

    return loss


@nb.njit(nogil=True)
def bayes_result(
    loss: nb.float64[:, :], minlength: nb.int64, new_c_link: nb.float64[:]
) -> nb.float64[:]:
    result = np.zeros(minlength)

    for j in range(minlength):
        l = -loss[:, j]
        l -= l.max()
        l = np.exp(l)
        l[~np.isfinite(l)] = 0.0
        l /= np.sum(l)
        result[j] = np.sum(l * new_c_link)

    return result


def calc_parameters_nbinom_c(
    y, mu, c_link, binnumbers, minlength, gamma, bayes, new_c_link
):
    loss = compute_2d_loss(y, mu, c_link, binnumbers, minlength, gamma, new_c_link)

    if bayes:
        result = bayes_result(loss, minlength, new_c_link)

    else:
        result = new_c_link[np.argmin(loss, axis=0)]

    return result
