
import abc
import logging
import warnings

import numexpr
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray
import scipy.special

from sklearn import base as sklearnb

from skpro.libs.cyclic_boosting import common_smoothers, learning_rate, link
from skpro.libs.cyclic_boosting.binning import get_feature_column_names_or_indices
from skpro.libs.cyclic_boosting.common_smoothers import SmootherChoice
from skpro.libs.cyclic_boosting.features import create_features, Feature, FeatureList, FeatureTypes, create_feature_id
from skpro.libs.cyclic_boosting.link import IdentityLinkMixin, LogLinkMixin
from skpro.libs.cyclic_boosting.utils import (
    slice_finite_semi_positive,
    nans,
    get_X_column,
    ConvergenceError,
    ConvergenceParameters,
)

from skpro.libs.cyclic_boosting.utils import get_normalized_values

from typing import Any, Dict, Optional, Union, Tuple, List, Set

_logger = logging.getLogger(__name__)


def get_influence_category(feature: Feature, influence_categories: dict) -> Union[None, str]:
    influence_category = None
    if influence_categories is not None:
        influence_category = influence_categories.get(feature.feature_group, None)
        if (
            influence_category is None
        ):  # this is due to the flaws in create features, i would propose to only allow tuple
            if len(feature.feature_group) == 1:
                fg = feature.feature_group[0]
                influence_category = influence_categories.get(fg, None)
        if influence_category is None:
            raise KeyError(f"Please add {feature.feature_group} to influence_categories")
    return influence_category


class UpdateMixin(object):
    def __init__(self, df: Optional[pd.DataFrame] = None, price_feature_seen: Optional[bool] = None):
        self.df = df
        self.price_feature_seen = price_feature_seen

    def include_price_contrib(self, pred: np.ndarray) -> None:
        self.df["exponents"] += pred
        self.price_feature_seen = True

    def include_factor_contrib(self, pred: np.ndarray, influence_category: Union[None, str]) -> None:
        self.df["factors"] += pred
        if influence_category is not None:
            if influence_category in self.df.columns:
                self.df[influence_category] += pred
            else:
                self.df[influence_category] = pred

    def remove_price_contrib(self, pred) -> None:
        self.df["exponents"] -= pred

    def remove_factor_contrib(self, pred: np.ndarray, influence_category: Union[None, str]) -> None:
        self.df["factors"] -= pred
        if influence_category is not None:
            if influence_category in self.df.columns:
                self.df[influence_category] -= pred
            else:
                self.df[influence_category] = pred

    def update_predictions(
        self, pred: np.ndarray, feature: Feature, influence_categories: Optional[dict] = None
    ) -> None:
        if feature.feature_type == FeatureTypes.external:
            self.include_price_contrib(pred)
        else:
            self.include_factor_contrib(pred, get_influence_category(feature, influence_categories))

    def remove_predictions(
        self, pred: np.ndarray, feature: Feature, influence_categories: Optional[dict] = None
    ) -> None:
        if feature.feature_type == FeatureTypes.external:
            self.remove_price_contrib(pred)
        else:
            self.remove_factor_contrib(pred, get_influence_category(feature, influence_categories))


class CBLinkPredictionsFactors(UpdateMixin):
    """Support for prediction of type log(p) = factors"""

    def __init__(self, predictions: np.ndarray):
        super().__init__(df=pd.DataFrame({"factors": predictions}))

    def predict_link(self) -> Union[ExtensionArray, np.ndarray]:
        return self.factors()

    def factors(self) -> Union[ExtensionArray, np.ndarray]:
        return self.df["factors"].values


class ZeroSmoother(sklearnb.BaseEstimator, sklearnb.RegressorMixin):
    def fit(self, X_for_smoother: np.ndarray, y: np.ndarray) -> Union[sklearnb.BaseEstimator, sklearnb.RegressorMixin]:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.repeat(0.0, len(X))


def _predict_factors(feature: Feature, X_for_smoother: np.ndarray, neutral_factor: float) -> np.ndarray:
    isfinite_all = slice_finite_semi_positive(X_for_smoother)
    prediction_smoother = nans(len(X_for_smoother))

    if isfinite_all.any() and (feature.smoother is not None):
        prediction_smoother[isfinite_all] = feature.learn_rate * feature.smoother.predict(X_for_smoother[isfinite_all])

    prediction_smoother[~isfinite_all] = feature.factors_link[-1]
    prediction_smoother[~np.isfinite(prediction_smoother)] = neutral_factor

    return prediction_smoother


def _factors_deviation(features: FeatureList) -> float:
    """Calculate mean absolute deviation of all factors.

    Parameters
    ----------

    features: :class:`~cyclic_boosting.base.FeatureList`
        Collection of all features.
    """
    mean_factor_devs = [np.mean(np.abs(feature.factors_link - feature.factors_link_old)) for feature in features]
    return np.mean(mean_factor_devs)



class CyclicBoostingBase(
    link.LinkFunction,
    sklearnb.BaseEstimator,
    metaclass=abc.ABCMeta,
):
    r"""The ``CyclicBoostingBase`` class implements the core skeleton for all
    Cyclic Boosting algorithms. It learns parameters (e.g., factors) for each
    feature-bin combination and applies smoothing algorithms (supplied by the
    ``smoother`` argument) on the parameters.

    By virtue of the ``feature_group`` parameter, each
    :mod:`cyclic_boosting` estimator can consider one- or higher-dimensional
    feature combinations.

    The parameter estimates are iteratively optimized until one of the stop
    criteria (``min_loss_change``, ``minimal_factor_change``,
    ``maximal_iterations``) is reached.

    Parameters
    ----------

    feature_groups: sequence of column labels
        (:obj:`str` or :obj:`int`) or tuples of such labels or
        :class:`cyclic_boosting.base.FeatureID`.
        For each feature or feature tuple in the sequence, a
        one- or multidimensional factor profile will be determined,
        respectively, and used in the prediction.

        If this argument is omitted, all columns except a possible
        ``weight_column`` are considered as one-dimensional feature_groups.

    hierarchical_feature_groups: sequence of column labels
        (:obj:`str` or :obj:`int`) or tuples of such labels or
        :class:`cyclic_boosting.base.FeatureID`.
        In the first three iterations of the training, only the feature groups
        defined here are used, i.e., all other feature groups are excluded.
        From the fourth iteration onwards, all feature groups are used. The
        idea of such hierarchical iterations is to support the modeling of
        hierarchical or causal effects (e.g., mitigate confounding).

        If this argument is not explicitly set, no such hierarchical iterations
        are run.

    feature_properties: :obj:`dict` of :obj:`int`
        Dictionary listing the names of all features for the training as keys
        and their pre-processing flags as values. When using a numpy feature
        matrix X with no column names the keys of the feature properties are
        the column indices.

        By default, ``flags.HAS_MISSING`` is assumed.
        If ``flags.MISSING_NOT_LEARNED`` is set for a feature group,
        the neutral factor 1 is used for all non-finite values.

    weight_column: string or int
        Column name or index used to include a weight column presented in X.
        If weight_column is ``None`` (default) equal weights for all samples
        are used.

    prior_prediction_column: string, int or None
        Column name or index used to include prior predictions. Instead of
        the target mean (``prior_prediction_column=None``), the predictions
        found in the ``prior_prediction_column`` column are used as a base
        model in :meth:`fit` and :meth:`predict`.

    minimal_loss_change: float
        Stop if the relative loss change of current
        and previous prediction falls below this value.

    minimal_factor_change: float
        Stop if the mean of the factor change falls below this value. The
        mean of the factor change is the mean of all absolute factor
        differences between the current and the previous iteration for each
        factor.

    maximal_iterations: int
        Maximal number of iteration.

    observers : list
        list of observer objects from the
        :mod:`~cyclic_boosting.observers` module.

    smoother_choice: subclass of :class:`cyclic_boosting.SmootherChoice`
        Selects smoothers

    output_column: string
        In case of the usage as a transformer, this column is added and
        filled with the predictions

    learn_rate: Functor or None
        Functor that defines the learning rate of each cyclic boosting iteration.
        It has to satisfy the interface of type :func:`learning_rate.constant_learn_rate_one`.
        If None is specified, which is the default, the learning rate is always 1.

    aggregate: boolean or None
        Description is required

    Notes
    -----

    **Preconditions**

    * Non-negative integer values starting at 0 **without gaps** are expected
      in each feature column.
    * The number of unique values in each feature column ``n_unique_feature``
      should be much smaller than the number of samples ``n``.

    Features transformed by
    :class:`cyclic_boosting.binning.BinNumberTransformer` satisfy these
    preconditions.

    * The target ``y`` has to be positive semi-definite.

    Attributes
    ----------

    global_scale_link_ : float
       The global scale of the target ``y`` in the linked space.

    stop_criteria_ : tuple
        tuple of three stop criteria are satisfied:
        ``(stop_iterations, stop_factor_change, stop_loss_change)``
    """
    supports_pandas = True
    inc_fitting = False
    no_deepcopy = {"feature_properties"}

    def __init__(
        self,
        feature_groups=None,
        hierarchical_feature_groups=None,
        training_iterations_hierarchical_features=3,
        feature_properties: Optional[Dict[int, int]] = None,
        weight_column: Optional[Union[str, int, None]] = None,
        prior_prediction_column: Optional[Union[str, int, None]] = None,
        minimal_loss_change: Optional[float] = 1e-6,
        minimal_factor_change: Optional[float] = 1e-4,
        maximal_iterations: Optional[int] = 10,
        observers: Optional[list] = None,
        smoother_choice: Optional[SmootherChoice] = None,
        output_column: Optional[str] = None,
        learn_rate: Optional[float] = None,
        aggregate: Optional[bool] = True,
    ):
        if smoother_choice is None:
            self.smoother_choice = common_smoothers.SmootherChoiceWeightedMean()
        else:
            self.smoother_choice = smoother_choice
            if not isinstance(smoother_choice, common_smoothers.SmootherChoice):
                raise ValueError("smoother_choice needs to be of type SmootherChoice")

        self.feature_groups = feature_groups
        self.hierarchical_feature_groups = hierarchical_feature_groups
        self.feature_properties = feature_properties

        self.features = None
        self.hierarchical_features = []
        if self.hierarchical_feature_groups is not None:
            for fg in self.hierarchical_feature_groups:
                hierarchical_feature = create_feature_id(fg)
                self.hierarchical_features.append(hierarchical_feature.feature_group)
        self.training_iterations_hierarchical_features = training_iterations_hierarchical_features
        self.feature_importances = {}
        self.aggregate = aggregate

        self.weight_column = weight_column
        self.weights = None
        self.prior_prediction_column = prior_prediction_column
        self.prior_pred_link_offset_ = 0
        self.global_scale_link_ = None

        self._check_parameters()
        self.minimal_loss_change = minimal_loss_change
        self.minimal_factor_change = minimal_factor_change
        self.maximal_iterations = maximal_iterations
        self.observers = observers
        if self.observers is None:
            self.observers = []

        self.iteration_ = None
        self.insample_loss_ = None
        self.stop_criteria_ = None
        self.neutral_factor_link = 0.0
        self.output_column = output_column
        if learn_rate is None:
            self.learn_rate = learning_rate.half_linear_learn_rate
        else:
            self.learn_rate = learn_rate
        if hierarchical_feature_groups is not None:
            self.learn_rate = learning_rate.constant_learn_rate_one
        self._init_features()

    def loss(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if not len(y) > 0:
            raise ValueError("Loss cannot be computed on empty data")
        else:
            sum_weighted_error = numexpr.evaluate("sum((prediction - y) * (prediction - y) * weights)")
            sum_weights = numexpr.evaluate("sum(weights)")
            return sum_weighted_error / sum_weights

    @property
    def global_scale_(self) -> np.ndarray:
        """
        Obtain the global scale in the space of the original target ``y``.
        """
        return self.unlink_func(self.global_scale_link_)

    def learning_rate(self, feature: Feature) -> float:
        lr = self.learn_rate(self.iteration_ + 1, self.maximal_iterations, feature)
        return lr

    def _init_weight_column(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """Sets the ``weights``.

        If a ``weight_column`` is given, it is retrieved
        from ``X``, else all weights are set to one.

        Parameters
        ----------

        X: np.ndarray
            samples features matrix
        """
        if self.weight_column is None:
            self.weights = np.ones(len(X))
        else:
            self.weights = get_X_column(X, self.weight_column)

    def _init_external_column(self, X: Union[pd.DataFrame, np.ndarray], is_fit: bool) -> None:
        """Init the external column.

        Parameters
        ----------

        X: np.ndarray
            samples features matrix
        is_fit: bool
           are we called in fit() ?
        """

    def _init_default_feature_groups(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Initializes the ``feature_groups`` to be a list of all
        columns available in ``X`` except the ``prior_prediction_column`` and
        the ``weight_column``.
        """
        exclude_columns = []
        if self.prior_prediction_column is not None:
            exclude_columns.append(self.prior_prediction_column)
        if self.weight_column is not None:
            exclude_columns.append(self.weight_column)

        self.feature_groups = get_feature_column_names_or_indices(X, exclude_columns=exclude_columns)

    def _init_global_scale(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> None:
        """
        Initializes the ``global_scale_link_`` and the
        ``prior_pred_link_offset_``. The ``global_scale_link_`` is set to the
        weighted mean of the target ``y`` transformed into the link space.
        The ``prior_pred_link_offset_`` is the difference between the
        ``global_scale_link_`` and the weighted mean of the prior prediction
        transformed to the link space.
        """
        if self.weights is None:
            raise RuntimeError("The weights have to be initialized.")

        minimal_prediction = 0.001
        self.global_scale_link_ = self.link_func(np.sum(y * self.weights) / np.sum(self.weights) + minimal_prediction)

        if self.prior_prediction_column is not None:
            prior_pred = get_X_column(X, self.prior_prediction_column)
            finite = np.isfinite(prior_pred)

            if not np.all(finite):
                _logger.warning(
                    "Found a total number of {} non-finite values in the prior prediction column".format(
                        np.sum(~finite)
                    )
                )

            prior_pred_mean = np.sum(prior_pred[finite] * self.weights[finite]) / np.sum(self.weights[finite])

            prior_pred_link_mean = self.link_func(prior_pred_mean)

            if np.isfinite(prior_pred_link_mean):
                self.prior_pred_link_offset_ = self.global_scale_link_ - prior_pred_link_mean
            else:
                warnings.warn(
                    "The mean prior prediction in link-space is not finite. "
                    "Therefore no individualization is done "
                    "and no prior mean subtraction is necessary."
                )
                self.prior_pred_link_offset_ = float(self.global_scale_link_)

    def _check_weights(self) -> None:
        if self.weights is None:
            raise RuntimeError("The weights have to be initialized.")

    def _init_features(self) -> None:
        """
        Initializes the ``features`` and binds the data belonging to a feature
        to it.
        """
        self.features = create_features(self.feature_groups, self.feature_properties, self.smoother_choice)

    def _get_prior_predictions(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.prior_prediction_column is None:
            prior_prediction_link = np.repeat(self.global_scale_link_, X.shape[0])
        else:
            prior_pred = get_X_column(X, self.prior_prediction_column)
            prior_prediction_link = self.link_func(prior_pred) + self.prior_pred_link_offset_
            finite = np.isfinite(prior_prediction_link)
            prior_prediction_link[~finite] = self.global_scale_link_

        return prior_prediction_link

    def _feature_contribution(self, feature: Feature, contribution: np.ndarray) -> np.ndarray:
        """The contribution of a feature group to the full model in link space.

        The link spaces have been chosen such that contributions in link space
        are always additive.

        Parameters
        ----------

        feature: :class:`Feature`
            feature providing the contribution

        contribution: ndarray
            the smoothed values in link space for input samples and the given
            feature group

        Returns
        -------

        ndarray
            the contribution to the full model in link space;
        """
        return contribution

    def is_diverged(self, loss) -> bool:
        return loss > self.initial_loss_ * 2

    def _check_convergence(self) -> None:
        msg = (
            "Your cyclic boosting training seems to be "
            "diverging. In the {0}. iteration the "
            "current loss: {1}, is greater "
            "than the trivial loss with just mean "
            "predictions: {2}.".format(self.iteration_ + 1, self.insample_loss_, self.initial_loss_)
        )

        if self.is_diverged(self.insample_loss_):
            self.is_diverging = True
            self.diverging += 1
            warnings.warn(msg)
            if self.diverging >= 5:
                raise ConvergenceError(msg)
        else:
            self.is_diverging = False

    def _update_loss(self, prediction: np.ndarray, y: np.ndarray) -> float:
        insample_loss_old = self.insample_loss_

        self.insample_loss_ = self.loss(prediction, y, self.weights)

        self.insample_msd_ = self.insample_loss_

        loss_change = insample_loss_old - self.insample_loss_
        self._check_convergence()

        if insample_loss_old != 0:
            loss_change /= insample_loss_old

        return loss_change

    def _log_iteration_info(self, convergence_parameters: ConvergenceParameters) -> None:
        if self.iteration_ == 0:
            _logger.info("Factor model iteration 1")
        else:
            _logger.info(
                "Factor model iteration: {iteration}, "
                "factors_deviation = {deviation}, "
                "loss_change = {loss_change}".format(
                    iteration=self.iteration_ + 1,
                    loss_change=convergence_parameters.loss_change,
                    deviation=convergence_parameters.delta,
                )
            )

    def _call_observe_feature_iterations(
        self, iteration: int, i: int, X: np.ndarray, y: np.ndarray, prediction: np.ndarray
    ) -> None:
        for observer in self.observers:
            observer.observe_feature_iterations(iteration, i, X, y, prediction, self.weights, self.get_state())

    def _call_observe_iterations(self, iteration, X, y, prediction, delta) -> None:
        for observer in self.observers:
            observer.observe_iterations(iteration, X, y, prediction, self.weights, self.get_state(), delta)

    def get_state(self) -> Dict[str, Any]:
        est_state = {
            "link_function": self,
            "features": self.features,
            "globale_scale": self.global_scale_,
            "insample_loss": self.insample_loss_,
        }
        if (
            self.hierarchical_feature_groups is not None
            and self.iteration_ < self.training_iterations_hierarchical_features
        ):
            est_state["features"] = [
                feature for feature in self.features if feature.feature_group in self.hierarchical_features
            ]
        return est_state

    def remove_preds(self, pred: CBLinkPredictionsFactors, X: np.ndarray) -> None:
        for feature in self.features:
            if feature.feature_type is None:
                weights = self.weights
            else:
                weights = self.weights_external

            feature.bind_data(X, weights)
            feature_predictions = self._pred_feature(X, feature, True)

            pred.remove_predictions(feature_predictions, feature)

            feature.fitted_aggregated -= feature.factors_link
            feature.unbind_data()

    def _set_factors_per_feature(self, feature: Feature) -> Tuple[float, int]:
        if feature.feature_type is not None:
            if len(feature.feature_group) == 1:
                min_f = 0.05
                max_f = 20
            else:
                min_f = 0.1
                max_f = 5
        else:
            if len(feature.feature_group) == 1:
                min_f = 0.01
                max_f = 1000
            else:
                min_f = 0.1
                max_f = 100
        return min_f, max_f

    def visit_factors(
        self, feature: Feature, unfitted_factors, X: np.ndarray, y: np.ndarray, pred: CBLinkPredictionsFactors
    ) -> None:
        clip = isinstance(self, LogLinkMixin)

        if clip and self.aggregate:
            # Maximal/Minimal updates
            feature.factors_link = np.clip(feature.factors_link, np.log(0.5), np.log(2), out=feature.factors_link)

            # Absolute size of factors/exponents per feature
            if feature.is_fitted:
                min_f, max_f = self._set_factors_per_feature(feature=feature)

                feature.factors_link = np.clip(
                    feature.factors_link,
                    np.log(min_f) - feature.fitted_aggregated,
                    np.log(max_f) - feature.fitted_aggregated,
                    out=feature.factors_link,
                )

            if self.is_diverging:

                def check_diverged() -> bool:
                    feature_predictions = self._pred_feature(X, feature, True)
                    pred.update_predictions(feature_predictions, feature)

                    prediction = self.unlink_func(pred.predict_link())

                    loss = self.loss(prediction, y, self.weights)
                    pred.remove_predictions(feature_predictions, feature)

                    return self.is_diverged(loss)

                if self.diverging > 1:
                    if check_diverged():
                        msg = "Feature factors for {} will be ignored due to diverging loss".format(
                            feature.feature_group
                        )
                        warnings.warn(msg)
                        feature.factors_link = -feature.fitted_aggregated
                        feature.smoother = ZeroSmoother()
                elif check_diverged():
                    msg = "Feature factors for {} will be clipped due to diverging loss".format(feature.feature_group)
                    warnings.warn(msg)
                    np.clip(
                        feature.factors_link,
                        unfitted_factors + np.log(0.8),
                        unfitted_factors + np.log(1.2),
                        out=feature.factors_link,
                    )

        if feature.is_fitted:
            if self.aggregate:
                feature.fitted_aggregated += feature.factors_link
            else:
                feature.fitted_aggregated = feature.factors_link.copy()
        else:
            feature.fitted_aggregated = feature.factors_link.copy()
            feature.is_fitted = True

    def feature_iteration(
        self, X: np.ndarray, y: np.ndarray, feature: Feature, pred: CBLinkPredictionsFactors, prefit_data
    ):
        if not self.aggregate:
            feature_predictions = self._pred_feature(X, feature, True)
            pred.remove_predictions(feature_predictions, feature)

        factors_link, uncertainties_link = self.calc_parameters(feature, y, pred, prefit_data=prefit_data)

        X_for_smoother = feature.update_factors(
            factors_link.copy(),
            uncertainties_link,
            self.neutral_factor_link,
            self.learning_rate(feature),
        )

        feature.factors_link = _predict_factors(
            feature=feature, X_for_smoother=X_for_smoother[:, : feature.dim], neutral_factor=self.neutral_factor_link
        )

        feature.factors_link = self.calibrate_to_weighted_mean(feature)

        self.visit_factors(feature, factors_link, X, y, pred)
        feature_predictions = self._pred_feature(X, feature, is_fit=True)
        pred.update_predictions(feature_predictions, feature)

        return pred

    def cb_features(
        self, X: np.ndarray, y: np.ndarray, pred: CBLinkPredictionsFactors, prefit_data
    ) -> Tuple[int, Any, Any]:
        for i, feature in enumerate(self.features):
            if feature.feature_type is None:
                weights = self.weights
            else:
                weights = self.weights_external
            if self.iteration_ == 0:
                feature.bind_data(X, weights)
                feature.factors_link = np.ones(feature.n_bins) * self.neutral_factor_link
                if prefit_data is not None:
                    prefit_data[i] = self.precalc_parameters(feature, y, pred)
            else:
                feature.bind_data(X, weights)
            yield i, feature, prefit_data[i]

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None
    ) -> Union[link.LinkFunction, sklearnb.BaseEstimator]:
        _ = self._fit_predict(X, y)
        return self

    def _init_fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._check_len_data(X, y)
        self._check_y(y)
        self._init_weight_column(X)
        self._init_external_column(X, True)

        if self.feature_groups is None:
            self._init_default_feature_groups(X)

        self._check_weights()
        self._init_features()
        self._init_global_scale(X, y)

    def _fit_main(self, X: np.ndarray, y: np.ndarray, pred: CBLinkPredictionsFactors) -> np.ndarray:
        self.diverging = 0
        self.is_diverging = False

        _logger.info("Cyclic Boosting global scale {}".format(self.global_scale_))

        prediction = self.unlink_func(pred.predict_link())

        self.insample_loss_ = self.loss(prediction, y, self.weights)
        self.initial_loss_ = self.insample_loss_
        self.initial_msd_ = self.insample_loss_
        self.iteration_ = 0
        prefit_data = [None for _ in self.features]

        convergence_parameters = ConvergenceParameters()

        while (not self._check_stop_criteria(self.iteration_, convergence_parameters)) or self.is_diverging:
            self._call_observe_iterations(self.iteration_, X, y, prediction, convergence_parameters.delta)

            self._log_iteration_info(convergence_parameters)
            for i, feature, pf_data in self.cb_features(X, y, pred, prefit_data):
                if (
                    self.hierarchical_feature_groups is not None
                    and self.iteration_ < self.training_iterations_hierarchical_features
                    and feature.feature_group not in self.hierarchical_features
                ):
                    feature.factors_link_old = feature.factors_link.copy()
                    continue
                pred = self.feature_iteration(X, y, feature, pred, pf_data)
                self._call_observe_feature_iterations(self.iteration_, i, X, y, prediction)

                if feature.factor_sum is None:
                    feature.factor_sum = [np.sum(np.abs(feature.fitted_aggregated))]
                else:
                    feature.factor_sum.append(np.sum(np.abs(feature.fitted_aggregated)))

            prediction = self.unlink_func(pred.predict_link())

            updated_loss_change = self._update_loss(prediction, y)
            convergence_parameters.set_loss_change(updated_loss_change=updated_loss_change)

            updated_delta = _factors_deviation(self.features)
            convergence_parameters.set_delta(updated_delta=updated_delta)

            if self.is_diverging:
                self.remove_preds(pred, X)

            self.iteration_ += 1

        # compute feature importances
        self.set_feature_importances()

        if len(self.observers) > 0:
            self.prepare_plots(X, y, prediction)

        self._call_observe_iterations(-1, X, y, prediction, convergence_parameters.delta)
        self._check_convergence()

        _logger.info("Cyclic Boosting, final global scale {}".format(self.global_scale_))

        for feature in self.features:
            feature.clear_feature_reference(observers=self.observers)

        return prediction

    def prepare_plots(self, X: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> None:
        for feature in self.features:
            if feature.feature_type is None:
                weights = self.weights
            else:
                weights = self.weights_external

            feature.bind_data(X, weights)

            sum_w, sum_yw, sum_pw = (
                np.bincount(feature.lex_binned_data, weights=w, minlength=feature.n_bins)
                for w in [weights, weights * y, weights * prediction]
            )

            mean_target_binned = (sum_yw + 1) / (sum_w + 1)

            if feature.n_bins > 1:
                mean_y_finite = np.sum(sum_yw[:-1]) / np.sum(sum_w[:-1])
                mean_prediction_finite = np.sum(sum_pw[:-1]) / np.sum(sum_w[:-1])
            else:
                mean_y_finite = np.mean(y)
                mean_prediction_finite = np.mean(prediction)

            mean_prediction_binned = (sum_pw + 1) / (sum_w + 1)

            if isinstance(self, IdentityLinkMixin):
                feature.mean_dev = mean_prediction_binned - mean_target_binned
                feature.y = mean_target_binned - mean_y_finite
                feature.prediction = mean_prediction_binned - mean_prediction_finite
            else:
                feature.mean_dev = np.log(mean_prediction_binned + 1e-12) - np.log(mean_target_binned + 1e-12)
                feature.y = np.log(mean_target_binned / mean_y_finite + 1e-12)
                feature.prediction = np.log(mean_prediction_binned / mean_prediction_finite + 1e-12)

            feature.y_finite = mean_y_finite
            feature.prediction_finite = mean_prediction_finite

            feature.learn_rate = 1.0

    def _fit_predict(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the estimator to the training samples.

        Iterate to calculate the factors and the global scale.
        """
        y = np.asarray(y)
        self._init_fit(X, y)
        pred = CBLinkPredictionsFactors(self._get_prior_predictions(X))
        prediction = self._fit_main(X, y, pred)

        del self.weights

        return prediction

    def _pred_feature(self, X: Union[pd.DataFrame, np.ndarray], feature: Feature, is_fit: bool):
        if is_fit:
            return feature.factors_link[feature.lex_binned_data]
        else:
            X_for_predict = get_X_column(X, feature.feature_group, array_for_1_dim=False)
            pred = _predict_factors(feature, X_for_predict, self.neutral_factor_link)
            return pred

    def predict(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None) -> np.ndarray:
        pred = self.predict_extended(X, None)
        return self.unlink_func(pred.predict_link())

    def predict_extended(
        self, X: Union[pd.DataFrame, np.ndarray], influence_categories: Optional[dict] = None
    ) -> CBLinkPredictionsFactors:
        self._check_fitted()
        self._init_external_column(X, False)

        prediction_link = self._get_prior_predictions(X)
        pred = CBLinkPredictionsFactors(prediction_link)

        for feature in self.features:
            feature_predictions = self._pred_feature(X, feature, is_fit=False)
            pred.update_predictions(feature_predictions, feature, influence_categories)

        return pred

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        if not self.output_column:
            raise KeyError("output_column not defined")
        try:
            if self.output_column in X.columns:
                raise KeyError("""output_column "{}" exists already""".format(self.output_column))
        except AttributeError:
            raise TypeError("data needs to be a dataframe for the usage as a fit_transformer")
        X[self.output_column] = self._fit_predict(X, y)
        return X

    def transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        if not self.output_column:
            raise KeyError("output_column not defined")
        try:
            if self.output_column in X.columns:
                raise KeyError("""output_column "{}" exists already""".format(self.output_column))
        except AttributeError:
            raise TypeError("data needs to be a dataframe for the usage as a transformer")
        X[self.output_column] = self.predict(X=X, y=y)
        return X

    def _check_stop_criteria(self, iterations: int, convergence_parameters: ConvergenceParameters) -> bool:
        """
        Checks the stop criteria and returns True if at least one is satisfied.

        You can check the stop criteria in the estimated parameter
        `stop_criteria_`.

        :rtype: bool
        """
        stop_iterations = False
        stop_factor_change = False
        stop_loss_change = False
        veto_hierarchical = False

        delta = convergence_parameters.delta
        loss_change = convergence_parameters.loss_change

        if iterations >= self.maximal_iterations:
            _logger.info(
                "Cyclic Boosting stopped because the number of "
                "iterations reached the maximum of {}.".format(self.maximal_iterations)
            )
            stop_iterations = True

        if delta <= self.minimal_factor_change:
            _logger.info(
                "Cyclic Boosting stopped because the change of "
                "the factors {0} was lower than the "
                "required minimum {1}.".format(delta, self.minimal_factor_change)
            )
            stop_factor_change = True

        if abs(loss_change) <= self.minimal_loss_change:
            _logger.info(
                "Cyclic Boosting stopped because the change of "
                "the loss {0} was lower "
                "than the required minimum {1}.".format(loss_change, self.minimal_loss_change)
            )
            stop_loss_change = True

        if loss_change < 0:
            _logger.warning(
                "Encountered negative change of loss. "
                "This might not be a problem, as long as the "
                "model converges. Check the LOSS changes in the "
                "analysis plots."
            )

        if (
            iterations <= self.training_iterations_hierarchical_features
            and self.hierarchical_feature_groups is not None
        ):
            veto_hierarchical = True

        self.stop_criteria_ = (stop_iterations, stop_factor_change, stop_loss_change)
        return (stop_iterations or stop_factor_change or stop_loss_change) and not veto_hierarchical

    def _check_parameters(self) -> None:
        if self.feature_groups is not None and len(self.feature_groups) == 0:
            raise ValueError("Please add some elements to `feature_groups`")

    def _check_fitted(self) -> None:
        """Check if fit was called"""
        if self.features is None:
            raise ValueError("fit must be called first.")

    def _check_len_data(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> None:
        """Check that input arrays are not empty"""
        if len(X) == 0 | len(y) == 0:
            raise ValueError("Estimator should be applied on non-empty data")

    def _check_y(self, y: np.ndarray) -> None:
        """Check that y has the correct values. Has to be
        implemented by the child class.
        """
        raise NotImplementedError(
            "You have to implement the function _check_y" " to ensure your target has correct values."
        )

    def get_subestimators_as_items(self, prototypes=True) -> List[Tuple]:
        """
        For prototypes=False, the clones are indexed by the column name
        or index.
        """
        if prototypes:
            return []
        else:
            self._check_fitted()
            return [(feature.feature_id, feature.smoother) for feature in self.features]

    def set_feature_importances(self) -> None:
        for feature in self.features:
            feature.prepare_feature()
            feature.set_feature_bin_deviations_from_neutral(self.neutral_factor_link)
            self.feature_importances[feature] = feature.bin_weighted_average

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns the relative importance of each input feature.
        """
        if not self.feature_importances:
            raise ValueError("_fit_main has to be called first to compute the feature importance.")
        else:
            normalized_values = get_normalized_values(self.feature_importances.values())
            norm_feature_importances = {
                "_".join(feature.feature_id.feature_group): normalized_values[i]
                for i, feature in enumerate(self.feature_importances.keys())
            }
            return norm_feature_importances

    def get_feature_contributions(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """Returns the contributions of each feature for each individual
        prediction on a given data set, offering individual explainability of
        the model. That means the learned parameter, e.g., factor for
        multiplicative modes, of the corresponding feature bin is accessed.
        These contributions should be interpreted in respect to the neutral
        element of the mode, i.e., 1 for multiplicative and 0 for additive
        modes.

        Parameters
        ----------
        X: np.ndarray
            data set to be predicted, needs to include the features of the
            model

        Returns
        -------
        dict
            contributions from each feature for each sample of X
        """
        contribution = {}
        for feature in self.features:
            ft = "" if feature.feature_type is None else "_{}".format(feature.feature_type)
            fg = " ".join(feature.feature_group)
            contribution[fg + ft] = self.unlink_func(self._pred_feature(X, feature, is_fit=False))
        return contribution

    @abc.abstractmethod
    def calc_parameters(self, feature: Feature, y: np.ndarray, pred: CBLinkPredictionsFactors, prefit_data):
        """Calculates factors and uncertainties of the bins of a feature group
        in the original space (not the link space) and transforms them to the
        link space afterwards

        The factors and uncertainties cannot be determined in link space, not
        least because target values like 0 diverge in link spaces like `log`
        or `logit`.

        The main loop will call the smoothers on the results returned by this
        method. Smoothing is always done in link space.

        This function must be implemented by the subclass according to its
        statistical model.

        Parameters
        ----------
        feature: :class:`~.Feature`
            class containing all features
        y: np.ndarray
            target, truth
        pred
            (in-sample) predictions from all other features (excluding the one
            at hand)
        prefit_data
            data returned by :meth:`~.precalc_parameters` during fit

        Returns
        -------
        tuple
            This method must return a tuple of ``factors`` and
            ``uncertainties`` in the **link space**.
        """
        raise NotImplementedError("implement in subclass")

    @abc.abstractmethod
    def precalc_parameters(self, feature: Feature, y: np.ndarray, pred: CBLinkPredictionsFactors):
        """Calculations that are not  dependent on intermediate predictions. If
        these are not needed, return :obj:`None` in the subclass.

        Results returned by this method will be served to
        :meth:`factors_and_uncertainties` as the ``prefit_data`` argument.

        Parameters
        ----------
        feature: :class:`~.Feature`
            class containing all features
        y: np.ndarray
            target, truth
        pred
            (in-sample) predictions from all other features (excluding the one
            at hand)
        """
        return None

    def required_columns(self) -> Set:
        required_columns = set()

        if self.weight_column is not None:
            required_columns.add(self.weight_column)

        if self.prior_prediction_column is not None:
            required_columns.add(self.prior_prediction_column)

        for feature_id in self.features:
            for col in feature_id.feature_group:
                required_columns.add(col)

        return required_columns

    def calibrate_to_weighted_mean(self, feature: Feature) -> np.ndarray:
        return feature.factors_link


def _coefficient_for_gaussian_matching_by_quantiles(perc: float) -> float:
    """The coefficient in the linear system of equations for the Gaussian
    matching by quantiles

    For details, see :func:`gaussian_matching_by_quantiles`.
    """
    return np.sqrt(2) * scipy.special.erfinv(2 * perc - 1)


def gaussian_matching_by_quantiles(
    dist, link_func, perc1: Union[float, np.ndarray], perc2: Union[float, np.ndarray]
) -> Tuple[float, float]:
    r"""Gaussian matching of the distribution transformed by the link function
    by demanding the equality of two quantiles in link space

    Since each quantile of a distribution is transformed just by the link
    function itself (by contrast to the mean or variance of the distribution),
    the Gaussian matching is quite easy and general::

        link_func(dist.ppf(perc<i>)) = gaussian_quantile<i>

    The connection between a Gaussian quantile and the distribution parameters
    is:

    .. math::
        \text{gaussian\_quantile}_i = \mu + \sigma \sqrt{2} \text{erf}^{-1}(
        2 \cdot \text{perc}_i - 1)

    Applied to both ``perc<i>``, this leads to a system of two linear equations
    that can be solved for ``mu`` and ``sigma``.

    Parameters
    ----------

    dist: scipy.stats distribution object
        the assumed distribution in the original space, e.g. a
        ``scipy.stats.beta`` in the case of
        :class:`~cyclic_boosting.CBClassifier`; it is assumed that this
        distribution has been initialized with vectorized parameters with one
        value for each cyclic boosting bin

    link_func: callable
        the transformation function to link space, e.g. the method
        :meth:`~cyclic_boosting.link.LinkFunction.link_func` inherited in
        the cyclic boosting class

    perc1: float or ndarray of floats
        percentage (a fixed value or an individual value for each of the bins)
        of the first quantile to match

    perc2: float or ndarray of floats
        percentage (a fixed value or an individual value for each of the bins)
        of the second quantile to match; it must be different from perc1.

    Returns
    -------

    tuple
        mu, sigma of the matched Gaussian, each being a ndarray with one value
        for each bin

    Example
    -------

    As example, consider the case of a beta distribution with link function
    logit which is used in :mod:`cyclic_boosting.classification`. It compares
    the old and new Gaussian matching for different alphas and betas.

    Derivation of the transformation of the beta distribution to logit space:

    .. math::
        y = \text{logit}(x) = \log(x / (1 - x))

        \Leftrightarrow x = \frac{1}{1 + e^{-x}}

        \Rightarrow \frac{\mathrm{d} x}{\mathrm{d} y} =
        \frac{e^{-y}}{\left(1 + e^{-y}\right)^2}

        p_{\text{beta}'(\alpha, \beta)}(y)\,\mathrm{d} y
        = p_{\text{beta}(\alpha, \beta)}(x)\,\mathrm{d} x
        = p_{\text{beta}(\alpha, \beta)}(x(y))
        \,\frac{\mathrm{d} x}{\mathrm{d} y}\,\mathrm{d} y

        = p_{\text{beta}(\alpha, \beta)}\left(\left(1 + e^{-y}\right)^2\right)
        \frac{e^{-y}}{\left(1 + e^{-y}\right)^2}\,\mathrm{d} y
    """
    quant1_link: float = link_func(dist.ppf(perc1))
    quant2_link: float = link_func(dist.ppf(perc2))

    a1 = _coefficient_for_gaussian_matching_by_quantiles(perc1)
    a2 = _coefficient_for_gaussian_matching_by_quantiles(perc2)

    sigma = (quant1_link - quant2_link) / (a1 - a2)
    mu = quant1_link - sigma * a1

    return mu, sigma


def calc_factors_generic(
    lex_binnumbers: np.ndarray,
    w_x: np.ndarray,
    w: np.ndarray,
    w_x2: np.ndarray,
    external_weights: np.ndarray,
    minlength: int,
    x0: float,
    w0: float,
) -> Tuple[float, float]:
    r"""Generic calculation of factors and uncertainties

    It is always possible to reparametrise :math:`\chi^2`
    of the different cyclic boosting models to the following standard form

    .. math::
        \chi^2 =\sum_i w_i \cdot (x_i - \hat{x})^2

    where the sum goes over the samples in one specific feature bin and
    :math:`\hat{x}` is the parameter to be estimated.

    **Multiplicative Ansatz:**

    Here we have

    .. math::
        \chi^2=\sum_i v_i \cdot (y_i - \hat{x} \cdot p_i)^2 / \sigma^2_{y_i}
        =\sum_i v_i p_i^2 \cdot (y_i/p_i - \hat{x})^2 / \sigma^2_{y_i}

    where :math:`v_i` is the external weight, :math:`y_i` is the target,
    :math:`p_i` is the prediction without the contribution of this feature
    and :math:`\sigma^2_{y_i}` is the variance of the target.
    By setting

    .. math::
        x_i = y_i / p_i
    .. math::
        w_i = v_i \cdot p_i^2 / \sigma^2_{y_i}

    we get the standard form.

    **Sum Ansatz:**

    Here we have

    .. math::
        \chi^2=\sum_i v_i \cdot (y_i - (\hat{x} - p_i))^2 / \sigma^2_{y_i}
        =\sum_i v_i \cdot ((y_i - p_i) - \hat{x})^2 / \sigma^2_{y_i}

    By setting

    .. math::
        x_i = y_i - p_i
    .. math::
        w_i = v_i / \sigma^2_{y_i}

    we get the standard form.

    **Slope Ansatz:**

    Here we have

    .. math::
        \chi^2=\sum_i v_i \cdot (y_i -
        (\hat{x} \cdot u_i + p_i))^2 / \sigma^2_{y_i}
        =\sum_i v_i u_i^2 \cdot ((y_i - p_i)/u_i - \hat{x})^2 / \sigma^2_{y_i}

    where :math:`u_i` is the external column, e.g. a price ratio for dynamic
    pricing models. By setting

    .. math::
        x_i = (y_i- p_i) / u_i
    .. math::
        w_i = v_i \cdot u_i^2 / \sigma^2_{y_i}

    we get the standard form.

    All models above fit in the **general linear ansatz:**

    .. math::
        \chi^2=\sum_i v_i \cdot (y_i -
        (\hat{x} \cdot m_i + b_i))^2 / \sigma^2_{y_i}
        =\sum_i v_i m_i^2 \cdot ((y_i - b_i)/m_i - \hat{x})^2 / \sigma^2_{y_i}

    By setting

    .. math::
        x_i = (y_i - b_i) / m_i
    .. math::
        w_i = v_i \cdot m_i^2 / \sigma^2_{y_i}
            = \frac{v_i}{\sigma^2_{y_i} \cdot (\frac{d x_i}{d y_i})^2}

    we get the standard form.

    **Solution for the general linear ansatz:**

    If we solve for :math:`\hat{x}`, we get the formula for the weighted
    mean:

    .. math::
        \hat{x} = \frac{\sum_i w_i \cdot x_i} {\sum_i w_i}

    The variance of :math:`\hat{x}` is

    .. math::
        \sigma^2_{\hat{x}} =
        \sum_i \left(\frac{d \hat{x}}{d x_i}\right)^2 \cdot \sigma^2_{x_i}
        = \sum_i \left(\frac{w_i}{\sum_i w_i}\right)^2 \cdot \sigma^2_{x_i}

    with

    .. math::
        w_i \cdot \sigma^2_{x_i} = \frac{v_i}{\sigma^2_{y_i}
        \cdot (\frac{d x_i}{d y_i})^2} \cdot \sigma^2_{y_i}
        \cdot \left(\frac{d x_i}{d y_i}\right)^2 = v_i

    we can write the variance as

    .. math::
        \sigma_{\hat{x}} = \frac{\sum_i w_i \cdot v_i} {(\sum_i w_i)^2}

    **Solution for the general linear ansatz with a prior on:** :math:`\hat{x}`

    To handle bins with low statistics or empty bins we introduce
    a gaussian prior with precision :math:`w_0` and mean :math:`x_0`

    Our :math:`\chi^2` now has the following form:

    .. math::
        \chi^2 =\sum_i w_i \cdot (x_i - \hat{x})^2 + w_0 \cdot (\hat{x} - x_0)^2

    Solving for :math:`\hat{x}` gives

    .. math::
        \hat{x} = \frac{x_0 \cdot w_0 + \sum_i w_i \cdot x_i} {w_0 + \sum_i w_i}

    The variance of :math:`\hat{x}` is

    .. math::
        \sigma^2_{\hat{x}} =
        \left(\frac{d \hat{x}}{d x_0}\right)^2 \cdot \sigma^2_{x_0} +
        \sum_i \left(\frac{d \hat{x}}{d x_i}\right)^2 \cdot \sigma^2_{x_i}
        = \frac{w_0 + \sum_i w_i \cdot v_i} {(w_0 + \sum_i w_i)^2}

    Note
    ----

    The quadratic sum is split in summands because parts of it may not be
    finite (e.g. in slope estimation).


    Parameters
    ----------

    lex_binnumbers: :class:`numpy.ndarray`
        1-dimensional numpy array containing the bin numbers.
    w_x: :class:`numpy.ndarray`
        :math:`w_i \cdot x_i` for each array element i
    w: :class:`numpy.ndarray`
        :math:`w_i` for each array element i
    w_x2: :class:`numpy.ndarray`
        :math:`w_i \cdot x_i^2` for each array element i
    external_weights: :class:`numpy.ndarray`
        :math:`v_i` for each array element i
    minlength: int
        number of bins for this feature including the `nan` bin
    x0: float
        mean of the prior on the mean
    w0: float
        prior precision of the prior on the mean

    Returns
    -------
    tuple
        A tuple of ``factors`` and ``uncertainties``
        in the original space.
    """

    sum_w_x, sum_w, sum_vw, sum_w_x2 = (
        np.bincount(lex_binnumbers, weights=w, minlength=minlength) for w in [w_x, w, external_weights * w, w_x2]
    )

    sum_w_x += w0 * x0
    sum_w += w0
    sum_w_x2 += w0 * x0**2
    sum_vw += w0

    weighted_mean = sum_w_x / sum_w
    variance_weighted_mean = sum_vw / sum_w**2

    return weighted_mean, np.sqrt(variance_weighted_mean)


__all__ = [
    "_factors_deviation",
    "CyclicBoostingBase",
    "gaussian_matching_by_quantiles",
    "calc_factors_generic",
]
