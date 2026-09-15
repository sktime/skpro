"""
base module for abstract smoothers
"""

import logging
from enum import Enum

import numpy as np

from skpro.libs.cyclic_boosting import utils
from skpro.libs.cyclic_boosting.smoothing.base import AbstractBinSmoother, SetNBinsMixin

_logger = logging.getLogger(__name__)


class RegressionType(Enum):
    """Type of regression that is supported by the smoother.

    Three variants are supported:

    * discontinuous: It cannot be interpolated between the values seen
       by the smoother (e.g. the values are unordered labels).
       Therefore all values in predict are set to nan that are out of the
       bin boundries in the fit or where no fit events have been seen.
    * interpolating: Interpolation is possible between the values seen by
       the smoother. So only values above or below the bin boundries are
       critical. Therefore all values in predict are set to nan that
       are out of the bin boundries in the fit.
    * extrapolating: Extrapolation allows arbitrary values independent of
       the values seen in the fit. Therefore no restrictions apply.
    """

    discontinuous = "discontinuous"
    interpolating = "interpolating"
    extrapolating = "extrapolating"


def check_reg_type(reg_type):
    if (
        (reg_type == RegressionType.discontinuous)
        or (reg_type == RegressionType.interpolating)
        or (reg_type == RegressionType.extrapolating)
    ):
        return reg_type
    else:
        raise KeyError(
            "Only the following regression types are allowed:"
            "{}, {}, {}".format(
                RegressionType.discontinuous,
                RegressionType.interpolating,
                RegressionType.extrapolating,
            )
        )


def not_interpolating(reg_type):
    return check_reg_type(reg_type) == RegressionType.discontinuous


def not_extrapolating(reg_type):
    return check_reg_type(reg_type) != RegressionType.extrapolating


class NormalizationSmoother(AbstractBinSmoother):
    """Meta-smoother that normalizes the values for its subestimator.

    Parameters
    ----------

    smoother: :class:`AbstractBinSmoother`
        smoother used to fit and predict on the `normalized` data points.
    """

    def __init__(self, smoother):
        self.smoother = smoother

    def calc_norm(self, X_for_smoother, y):
        """Calculate the weighted mean of the target y that can
        then be used to give the subestimator a normalized
        distribution.

        Parameters
        ----------

        X_for_smoother: :class:`numpy.ndarray`
            Multidimensional array with at least three columns.
            For a k-dimensional feature the first k columns contain
            the x-values for the smoothing.
            The ``k + 1`` column are the weights of these x-values, while
            the ``k + 2`` column contains the uncertainties.

        smoothed_y: :class:`numpy.ndarray`
             Array that contains the original target values.
        """
        w = X_for_smoother[:, -2]
        weightsum = np.sum(w)
        if weightsum > 0:
            self.norm_ = np.sum(y * w) / weightsum
        else:
            self.norm_ = 0

        if not np.isfinite(self.norm_):
            _logger.info(
                "The norm in smoother {} is not finite: "
                "norm= {}; weights= {}; target= {}".format(
                    self.__class__.__name__, self.norm_, X_for_smoother, y
                )
            )
            self.norm_ = 0.0

    def fit(self, X_for_smoother, y):
        self.calc_norm(X_for_smoother, y)
        self.smoother.fit(X_for_smoother, y - self.norm_)

    def predict(self, X_for_smoother):
        smoothed_y = self.smoother.predict(X_for_smoother)
        return smoothed_y + self.norm_


def _selected_events_interpolating(X_for_smoother, ndim, nbins):
    selected_events = np.min(X_for_smoother[:, :ndim], axis=1) < 0.0
    for dim in range(ndim):
        selected_events |= X_for_smoother[:, dim] > (nbins[dim] - 0.5)
    return selected_events


class RegressionTypeSmoother(AbstractBinSmoother, SetNBinsMixin):
    """Meta-smoother to constrain all values according to their
    :class:`~cyclic_boosting.smoothing.RegressionType` from the ``predict`` method
    of the subsmoother.

    Parameters
    ----------

    smoother: :class:`AbstractBinSmoother`
        smoother used to fit and predict on the `normalized` data points.

    reg_type: :class:`RegressionType`
        defines the regression type that is used to constrain the values.

    Regression Types
    ----------------

    * discontinuous: Set all values in predict to nan that are out of the bin
           boundries in the fit or where no fit events have been seen.
    * interpolating: Set all values in predict to nan that are out of the bin
           boundries in the fit.
    * extrapolating: No restrictions for values in predict.
    """

    def __init__(self, smoother, reg_type):
        self.smoother = smoother
        self.reg_type = check_reg_type(reg_type)

    def apply_cut(self, X_for_smoother, smoothed_y):
        """Constrain all values in smoothed_y according to their
        :class:`RegressionType`.

        Parameters
        ----------

        X_for_smoother: :class:`numpy.ndarray`
            Multidimensional array with at least three columns.
            For a k-dimensional feature the first k columns contain
            the x-values for the smoothing.
            The ``k + 1`` column are the weights of these x-values, while
            the ``k + 2`` column contains the uncertainties.

        smoothed_y: :class:`numpy.ndarray`
             Array that contains the result of the subsmoother.
        """
        if not_interpolating(self.reg_type):
            selected_events = utils.not_seen_events(
                X_for_smoother[:, : self.ndim_], self.bin_weights_, self.n_bins_
            )
        elif not_extrapolating(self.reg_type):
            selected_events = _selected_events_interpolating(
                X_for_smoother, self.ndim_, self.n_bins_
            )
        else:
            return smoothed_y

        smoothed_y = np.asarray(smoothed_y, dtype=np.float64)
        smoothed_y[selected_events] = np.nan
        return smoothed_y

    def fit(self, X_for_smoother, y):
        self.set_n_bins(X_for_smoother)
        self.smoother.fit(X_for_smoother, y)

    def predict(self, X_for_smoother):
        smoothed_y = self.smoother.predict(X_for_smoother)
        return self.apply_cut(X_for_smoother, smoothed_y)


class NormalizationRegressionTypeSmoother(
    NormalizationSmoother, RegressionTypeSmoother
):
    """Meta-smoother to constrain all values according to their
    :class:`~cyclic_boosting.smoothing.RegressionType` from the ``predict`` method
    of the subsmoother.

    Parameters
    ----------

    smoother: :class:`AbstractBinSmoother`
        smoother used to fit and predict on the `normalized` data points.

    reg_type: :class:`RegressionType`
        defines the regression type that is used to constrain the values.

    Regression Types
    ----------------

    * discontinuous: Set all values in predict to nan that are out of the bin
           boundries in the fit or where no fit events have been seen.
    * interpolating: Set all values in predict to nan that are out of the bin
           boundries in the fit.
    * extrapolating: No restrictions for values in predict.
    """

    def __init__(self, smoother, reg_type):
        self.smoother = smoother
        self.reg_type = check_reg_type(reg_type)

    def fit(self, X_for_smoother, y):
        self.set_n_bins(X_for_smoother)
        self.calc_norm(X_for_smoother, y)
        self.smoother.fit(X_for_smoother, y - self.norm_)

    def predict(self, X_for_smoother):
        smoothed_y = self.smoother.predict(X_for_smoother) + self.norm_
        return self.apply_cut(X_for_smoother, smoothed_y)


class SectionSmoother(AbstractBinSmoother):
    """Meta-smoother that splits the fitted data into two parts which are
    fitted by different subsmoothers.

    Parameters
    ----------

    split_point : float
        value of x to split the data :math:`x_{below} <= C_{split} < x_{above}`

    smoother_lower: :class:`AbstractBinSmoother`
        smoother used to fit and predict on the data points below and including
        the split_point

    smoother_upper: :class:`AbstractBinSmoother`
        smoother used to fit and predict on the data points above the split_point

    nan_representation: float
        optional argument to define ``not a number values`` (default = ``np.nan``).

    epsilon: float
        Floating point accuracy when comparing with the ``split_point``.
        (E.g. :math:`x_{below} <= C_{split} + /epsilon`)
    """

    def __init__(
        self,
        split_point,
        smoother_lower,
        smoother_upper,
        nan_representation=np.nan,
        epsilon=0.001,
    ):
        self.split_point = split_point
        self.smoother_lower = smoother_lower
        self.smoother_upper = smoother_upper
        self.nan_representation = np.nan
        self.epsilon = epsilon
        self._reset_smoother_status()

    def _reset_smoother_status(self):
        self.smoother_lower_fitted = False
        self.smoother_upper_fitted = False

    def _split_condition(self, X_for_smoother):
        return X_for_smoother[:, 0] <= self.split_point + self.epsilon

    def fit(self, X_for_smoother, y):
        self._reset_smoother_status()
        cond_lower = self._split_condition(X_for_smoother)
        if np.sum(cond_lower) > 0:
            self.smoother_lower.fit(X_for_smoother[cond_lower], y[cond_lower])
            self.smoother_lower_fitted = True
        if np.sum(~cond_lower) > 0:
            self.smoother_upper.fit(X_for_smoother[~cond_lower], y[~cond_lower])
            self.smoother_upper_fitted = True
        assert self.smoother_upper_fitted or self.smoother_lower_fitted

    def predict(self, X_for_smoother):
        if not self.smoother_upper_fitted and not self.smoother_lower_fitted:
            raise ValueError(f"The {self.__class__.__name__} has not been fitted!")

        cond_lower = self._split_condition(X_for_smoother)
        pred = np.ones(len(X_for_smoother)) * self.nan_representation
        if self.smoother_lower_fitted and np.sum(cond_lower) > 0:
            pred[cond_lower] = self.smoother_lower.predict(X_for_smoother[cond_lower])
        if self.smoother_upper_fitted and np.sum(~cond_lower) > 0:
            pred[~cond_lower] = self.smoother_upper.predict(X_for_smoother[~cond_lower])
        return pred


__all__ = [
    "NormalizationSmoother",
    "RegressionTypeSmoother",
    "NormalizationRegressionTypeSmoother",
    "SectionSmoother",
]
