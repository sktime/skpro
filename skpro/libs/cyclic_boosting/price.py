# -*- coding: utf-8 -*-
"""
Cyclic boosting exponential price model estimator
"""

import logging

import numpy as np
import pandas as pd
from numexpr import evaluate


from skpro.libs.cyclic_boosting import CBNBinomRegressor
from skpro.libs.cyclic_boosting.features import FeatureTypes, create_feature_id
from skpro.libs.cyclic_boosting.base import UpdateMixin
from skpro.libs.cyclic_boosting.regression import _calc_factors_and_uncertainties
from skpro.libs.cyclic_boosting.utils import get_X_column

_logger = logging.getLogger(__name__)


def combine_lists_of_feature_groups(standard_feature_groups, external_feature_groups):
    """Combine the ``feature_groups`` and ``external_feature_groups`` to
    one list of feature_ids.
    """
    if standard_feature_groups is None:
        raise ValueError("Pass at least one feature in `standard_feature_groups`.")
    standard_feature_groups = [create_feature_id(fg) for fg in standard_feature_groups]
    if external_feature_groups is not None:
        external_feature_groups = [
            create_feature_id(exfg, default_type=FeatureTypes.external) for exfg in external_feature_groups
        ]
        _check_feature_groups(standard_feature_groups, external_feature_groups)
        return standard_feature_groups + external_feature_groups
    else:
        _check_feature_groups(standard_feature_groups, external_feature_groups)
        return standard_feature_groups


def _check_feature_groups(standard_feature_groups, external_feature_groups):
    """Check if ``standard_feature_groups`` and ``external_feature_groups``
    are correctly filled with the right
    :class:`cyclic_boosting.base.FeatureID`.
    """
    for feature_id in standard_feature_groups:
        if feature_id.feature_type is not None:
            raise ValueError(
                "There are none standard features in the "
                "``standard_feature_groups`` present. Use "
                "``external_feature_groups`` for those features."
            )
    if external_feature_groups is not None:
        for feature_id in external_feature_groups:
            if feature_id.feature_type is not FeatureTypes.external:
                raise ValueError(
                    "There are standard features in the "
                    "``external_feature_groups`` present. Use "
                    "``standard_feature_groups`` for those features."
                )


def _set_right_bound(x, args, valid):
    bounds_to_small = True
    max_iter = 10
    i = 0
    while bounds_to_small and (i < max_iter):
        f = newton_step(x, *args, only_jac=True)
        index = valid & (f < 0)
        bounds_to_small = np.any(index)
        if bounds_to_small:
            x[index] *= 2
        else:
            break
        i += 1
    return x


def newton_bisect(
    newton_step,
    args,
    valid,
    x_l,
    x_r,
    epsilon=0.001,
    maximal_iterations=10,
    start_values=None,
):
    """Combination of Newton's Method and Bisection
    to find the root of the jacobian.

    Every time Newton tries a step outside the bounding interval [x_l, x_r]
    the bisection update is used.

    Parameters
    ----------
    newton_step: callable
       Calculate the parameter update, jacobian and hessian:

       .. code-block:: python

           def newton_step(l, *args):
               ...
               return l_new, jacobian, hessian

    args: iterable
       Parameters pass to `newton_step`
    valid: np.ndarray
       Boolean array encoding valid values
    x_l: np.ndarray
       Start values for the left bounds
    x_r: np.ndarray
       Start values for the right bounds
    epsilon: float
       Epsilon on the jacobian. Iteration stops if:

       .. code-block:: python

           np.all(np.abs(jacobian) < epsilon)

    maximal_iterations: int
       number of maximal iterations
    start_values: np.ndarray or None
       Start values on the parameters. If None, the middle of
       the first bisection interval is used.

    Returns
    -------
    tuple
        (fitted parameters, variance of parameters,
             left bounds, right bounds)
    """
    x_r = _set_right_bound(x_r, args, valid)
    x_l_save, x_r_save = x_l.copy(), x_r.copy()

    if start_values is None or np.any(start_values < x_l) or np.any(start_values > x_r):
        l = (x_r + x_l) / 2.0
    else:
        l = start_values

    for i in range(maximal_iterations):
        l_new, jac, hess = newton_step(l, *args)
        finite = np.isfinite(jac) & np.isfinite(hess) & np.isfinite(l_new)
        valid = valid & finite
        converged = (np.abs(jac) < epsilon) | (np.abs(l - x_l) < epsilon) | (np.abs(x_r - l) < epsilon) | (~valid)
        if np.all(converged):
            break
        x_l[valid & (jac < 0)] = l[valid & (jac < 0)]
        x_r[valid & (jac >= 0)] = l[valid & (jac >= 0)]

        index = np.full_like(valid, False)
        index[valid] = (l_new[valid] > x_r[valid]) | (l_new[valid] < x_l[valid])

        l_new[index] = (x_l[index] + x_r[index]) / 2.0
        l_new[~valid] = 1.0
        hess[~valid] = 1.0
        l = l_new

    # Check for negative values in hessian -> not allowed because negative variance
    hess_inv = np.full_like(hess, np.inf)
    m_non_zero = hess != 0
    hess_inv[m_non_zero] = 1.0 / hess[m_non_zero]

    ind = hess_inv <= 0
    hess_inv[ind] = 1.0

    return l, hess_inv, x_l_save, x_r_save


def newton_step(
    l,
    y,
    p,
    log_x,
    k,
    prior,
    s,
    log_k_prior,
    var_l,
    lex_binnumbers,
    minlength,
    only_jac=False,
):
    l1_ = np.c_[l, np.log(l)]  # noqa
    l1_ = l1_[lex_binnumbers, :]  # noqa
    l1 = l1_[:, 0]  # noqa
    log_l1 = l1_[:, 1]  # noqa

    mu = evaluate("p * exp((k * l1) * log_x)")  # noqa
    w = k * log_x  # noqa

    bin_counts = np.bincount(lex_binnumbers, minlength=minlength)[lex_binnumbers]  # noqa
    # wolfram alpha: jacobian matrix ((y - p * x ** (l * k))^2 * s + ((l*k - c)^2 +
    # (l - 1)^2 + log(l)^2 + log(l*k / c)^2) / v) with respect to (l)
    jac_prior = evaluate("2 * (k * (k * l1 - prior) + l1 - 1 + (log_k_prior + 2 * log_l1) / l1) / var_l / bin_counts")

    jac_data = evaluate("- (2 * w * mu * (y - mu)) * s")
    jacobian = np.bincount(lex_binnumbers, weights=jac_data + jac_prior, minlength=minlength)

    if only_jac:
        return jacobian

    # wolfram alpha: hessian matrix ((y - p * x ** (l * k))^2 * s + ((l*k - c)^2 +
    # (l - 1)^2 + log(l)^2 + log(l*k / c)^2) / v) with respect to (l)
    hess_prior = evaluate("2 * (k * k + 1 + (2 - (log_k_prior + 2 * log_l1) ) / l1 / l1) / var_l / bin_counts")

    hess_data = evaluate("(2 * w * w * mu * mu) * s + w * jac_data")
    hessian = np.bincount(lex_binnumbers, weights=hess_data + hess_prior, minlength=minlength)

    lnew = np.full_like(l, np.nan)
    m_non_zero = hessian != 0
    lnew[m_non_zero] = l[m_non_zero] - jacobian[m_non_zero] / hessian[m_non_zero]

    return lnew, jacobian, hessian


class CBExponential(CBNBinomRegressor):
    def __init__(
        self,
        external_colname,
        standard_feature_groups,
        external_feature_groups,
        feature_properties=None,
        weight_column=None,
        prior_prediction_column=None,
        minimal_loss_change=1e-3,
        minimal_factor_change=1e-3,
        maximal_iterations=20,
        observers=None,
        var_prior_exponent=0.1,
        smoother_choice=None,
        prior_exponent_colname=None,
        output_column=None,
        learn_rate=None,
        a=1.0,
        c=0.0,
    ):
        self.standard_feature_groups = standard_feature_groups
        self.external_feature_groups = external_feature_groups
        self.prior_exponent_colname = prior_exponent_colname

        feature_id_list = combine_lists_of_feature_groups(standard_feature_groups, external_feature_groups)

        self.external_colname = external_colname

        CBNBinomRegressor.__init__(
            self,
            feature_groups=feature_id_list,
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
            a=a,
            c=c,
        )

        # Parameters which influence the exponent fits
        # in each feature group bin
        self.var_prior_exponent = var_prior_exponent
        self.epsilon_jacobian = 0.01
        self.starting_bound_bisect = 2

    def required_columns(self):
        required_columns = CBNBinomRegressor.required_columns(self)
        required_columns.add(self.external_colname)
        if self.prior_exponent_colname is not None:
            required_columns.add(self.prior_exponent_colname)
        return required_columns

    def _init_external_column(self, X, is_fit):
        """Init the external column.

        Parameters
        ----------

        X: np.ndarray
            samples features matrix
        is_fit: bool
           are we called in fit() ?

        Raises
        ------

        ValueError
            if any value in 'external_col' is smaller equal zero

        """
        self.external_col = get_X_column(X, self.external_colname) - 1
        is_finite = np.isfinite(self.external_col)
        self.external_col[~is_finite] = 0
        if is_fit:
            self.weights_external = np.asarray(is_finite, dtype=np.float64)
            self.weights_external *= self.weights

    def _get_prior_exponent(self, X):
        if self.prior_exponent_colname is None:
            return -1.0
        else:
            return get_X_column(X, self.prior_exponent_colname)

    def calc_parameters(self, feature, y, pred, prefit_data):
        """Calculates factors and uncertainties of the bins of a feature group
        in the original space (not the link space) and transforms them to the
        link space afterwards

        The factors and uncertainties cannot be determined in link space, not
        least because target values like 0 diverge in link spaces like `log`
        or `logit`.

        Parameters
        ----------
        feature: :class:`~.Feature`
            feature information
        y: np.ndarray
            target, truth
        prediction_link: np.ndarray
            prediction in link space of all *other* features.
        prefit_data
            data returned by
            :meth:`~cyclic_boosting.base.CyclicBoostingBase.precalc_parameters`
            during fit

        Returns
        -------
        tuple
            This method must return a tuple of ``factors`` and
            ``uncertainties`` in the **link space**.
        """
        if feature.feature_type == FeatureTypes.external:
            _logger.debug("Price Feature {}".format(feature.feature_group))
            expo, variance = self._calc_parameters_exponent(feature, y, pred)
            return expo, variance
        else:
            _logger.debug("Demand Feature {}".format(feature.feature_group))
            return CBNBinomRegressor.calc_parameters(self, feature, y, pred, None)

    def _calc_parameters_exponent(self, feature, y, pred):
        """Calculates exponents and uncertainties
        for each bin of a feature group.

        Parameters
        ----------
        feature: :class:`~.Feature`
            feature information
        y: np.ndarray
            target, truth
        prediction_link: np.ndarray
            prediction in link space of all *other* features.


        Returns
        -------
        tuple
            ``exponents`` and ``uncertainties``.
        """
        lex_binnumbers = feature.lex_binned_data
        pred_factors = self.unlink_func(pred.factors())
        pred_expos = pred.exponents()
        base = pred.base()
        prior_exponents = pred.prior_exponents()

        bounds_l = np.zeros(feature.n_bins, dtype=np.float64)
        feature.bounds_r = self.starting_bound_bisect * np.ones(feature.n_bins, dtype=np.float64)
        start_values = np.ones(feature.n_bins, dtype=np.float64)

        p = self.unlink_func(pred.predict_link())

        log_k_prior = evaluate("log(pred_expos / prior_exponents)")
        a = self.a
        c = self.c
        variance = a * p + c * p * p

        factors, variance_factors, feature.bounds_l, feature.bounds_r = newton_bisect(
            newton_step,
            (
                y,
                pred_factors,
                base,
                pred_expos,
                prior_exponents,
                self.weights_external / variance,
                log_k_prior,
                self.var_prior_exponent,
                lex_binnumbers,
                len(feature.bin_weightsums),
            ),
            feature.bin_weightsums > 0,
            x_l=bounds_l,
            x_r=feature.bounds_r,
            epsilon=self.epsilon_jacobian,
            start_values=start_values,
        )
        return gamma_momemt_matching(factors, variance_factors, self.link_func)

    def predict(self, X, y=None):
        pred = self.predict_extended(X, None)
        return self.unlink_func(pred.predict_link())

    def predict_extended(self, X, influence_categories):
        self._check_fitted()
        self._init_external_column(X, False)

        prediction_link = self._get_prior_predictions(X)
        pred = CBLinkPredictions(prediction_link, self._get_prior_exponent(X), self.external_col)

        for feature in self.features:
            feature_predictions = self._pred_feature(X, feature, is_fit=False)
            pred.update_predictions(feature_predictions, feature, influence_categories)
        return pred

    def fit(self, X, y=None):
        self._init_fit(X, y)
        pred = CBLinkPredictions(
            self._get_prior_predictions(X),
            self._get_prior_exponent(X),
            self.external_col,
        )
        _ = self._fit_main(X, y, pred)
        del self.external_col
        del self.weights_external

        return self


def gamma_momemt_matching(factors, variance_factors, link_func):
    """ """
    # Gamma distribution moment matching
    beta = factors / variance_factors
    alpha = beta * factors

    # check finite of alpha, beta. If not finite
    # -> set alpha=0, beta=0 -> then only the prior is used
    # Non-finite values should not happen anymore since a
    # prior is used in calc_factors_generic.
    is_finite = np.isfinite(beta) & np.isfinite(alpha)
    beta[~is_finite] = 0
    alpha[~is_finite] = 0

    return _calc_factors_and_uncertainties(alpha, beta, link_func)


class CBLinkPredictions(UpdateMixin):
    """Support for prediction of type log(p) = factors + base * exponents"""

    def __init__(self, predictions, exponents, base):
        self.df = pd.DataFrame(
            {
                "factors": predictions,
                "prior_exponents": exponents,
                "exponents": np.zeros_like(exponents, dtype=np.float64),
                "base": base,
            }
        )

    def predict_link(self):
        return self.factors() + self.exponents() * self.base()

    def factors(self):
        return self.df["factors"].values

    def exponents(self):
        x = self.df["exponents"].values  # noqa
        return self.df["prior_exponents"].values * evaluate("exp(x)")

    def prior_exponents(self):
        return self.df["prior_exponents"].values

    def base(self):
        return self.df["base"].values
