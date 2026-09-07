"""Convenient smoother mappings from feature property values (and tuples) to
typical smoothers used for the Cyclic Boosting regressor"""

import logging
from typing import Optional

from skpro.libs.cyclic_boosting import flags, smoothing
from skpro.libs.cyclic_boosting.smoothing.base import AbstractBinSmoother
from skpro.libs.cyclic_boosting.smoothing.meta_smoother import (
    NormalizationRegressionTypeSmoother,
    NormalizationSmoother,
    RegressionType,
    RegressionTypeSmoother,
)

_logger = logging.getLogger(__name__)


def _simplify_flags(feature_property: int, feature_group: Optional[str] = None):
    """
    Simplifies a general flag to a basic set to select a smoother later on.

    Parameters
    ----------
    feature_property:
        A single feature property.

    feature_group:
        Optional argument for the name of the feature/feature_group
        corresponding to the feature_property used for logging.
    """
    if flags.is_linear_set(feature_property):
        return flags.IS_LINEAR
    elif flags.is_seasonal_set(feature_property):
        return flags.IS_SEASONAL
    elif flags.is_monotonic_set(feature_property):
        if flags.increasing_set(feature_property):
            return flags.IS_MONOTONIC | flags.INCREASING
        elif flags.decreasing_set(feature_property):
            return flags.IS_MONOTONIC | flags.DECREASING
        else:
            return flags.IS_MONOTONIC
    elif flags.is_continuous_set(feature_property):
        return flags.IS_CONTINUOUS
    elif flags.is_ordered_set(feature_property):
        return flags.IS_ORDERED
    elif flags.is_unordered_set(feature_property):
        return flags.IS_UNORDERED
    else:
        features = ""
        if feature_group is not None:
            features = f"for feature {feature_group}"
        if flags.has_missing_set(feature_property):
            _logger.info(
                "No feature property set. Thus it is set to default IS_UNORDERED!"
            )
        else:
            _logger.warning(
                "Feature property {} is not known {}."
                " Thus it is converted to IS_UNORDERED!".format(
                    flags.flags_to_string(feature_property), features
                )
            )
        return flags.IS_UNORDERED


def _default_smoother_types(neutral_factor_link=0, use_normalization=True):
    smoother_types = {
        flags.IS_UNORDERED: smoothing.onedim.WeightedMeanSmoother(
            prior_prediction=neutral_factor_link
        ),
        flags.IS_ORDERED: smoothing.onedim.WeightedMeanSmootherNeighbors(),
        flags.IS_CONTINUOUS: smoothing.onedim.OrthogonalPolynomialSmoother(),
        flags.IS_LINEAR: smoothing.extrapolate.LinearExtrapolator(),
        flags.IS_SEASONAL:
        # the seasonal smoother does not work with offset_tozero
        # when the normalization is done within the smoother
        smoothing.onedim.SeasonalSmoother(offset_tozero=not use_normalization),
        flags.IS_MONOTONIC: smoothing.onedim.IsotonicRegressor(increasing="auto"),
        flags.IS_MONOTONIC
        | flags.INCREASING: smoothing.onedim.IsotonicRegressor(increasing=True),
        flags.IS_MONOTONIC
        | flags.DECREASING: smoothing.onedim.IsotonicRegressor(increasing=False),
    }
    return smoother_types


def determine_reg_type(feature_group, feature_property, feature_type):
    """Function to determine the RegressionType of a feature.

    Parameters
    ----------

    feature_group: tuple
        Name of the feature_group

    feature_property: tuple
        Tuple of feature properties.

    feature_type:
        Type of the feature.
    """
    if not isinstance(feature_property, tuple):
        if flags.is_linear_set(feature_property):
            return RegressionType.extrapolating
        elif flags.is_seasonal_set(feature_property) or flags.is_continuous_set(
            feature_property
        ):
            return RegressionType.interpolating
        else:
            return RegressionType.discontinuous
    else:
        reg_types = [
            determine_reg_type(fg, fp, feature_type)
            for fg, fp in zip(feature_group, feature_property)
        ]
        if RegressionType.discontinuous in reg_types:
            return RegressionType.discontinuous
        elif RegressionType.interpolating in reg_types:
            return RegressionType.interpolating
        else:
            return RegressionType.extrapolating


def determine_meta_smoother(smoother, use_normalization, reg_type=None):
    """Wrapper function that chooses the correct meta_estimators
    for the smoothers in the SmootherChoice class.

    Parameters
    ----------

    smoother: :class:`AbstractBinSmoother`
        smoother that should be wrapped.

    use_normalization: bool
        Flag to decide if normalization should be used for the smoothers.

    reg_type: :class:`RegressionType`
        If a ``RegressionType`` is set the RegressionTypeSmoother is used.
    """

    if not use_normalization and reg_type is None:
        return smoother
    if use_normalization and reg_type is None:
        return NormalizationSmoother(smoother)
    if not use_normalization and reg_type is not None:
        return RegressionTypeSmoother(smoother, reg_type)
    if use_normalization and reg_type is not None:
        return NormalizationRegressionTypeSmoother(smoother, reg_type)


class SmootherChoice:
    r"""Base class for selecting smoothers for cyclic boosting.

    Maps feature property tuples to smoothers in 1D/2D/3D.

    Parameters
    ----------

    use_regression_type: bool
        Flag to decide if ``RegressionType`` regularization should be used
        for the smoothers.
        (default = True)

    use_normalization: bool
        Flag to decide if normalization should be used for the smoothers.
        (default = True)

    explicit_smoothers: dict
        A dictionary with custom 1-d smoothers that override the default
        one-dimensional smoothers chosen by the feautre property.
        Needs to be of the format {feature_group : smoother},
        where feature_group is a tuple of strings and smoother is
        and instance of AbstractBinSmoother.
        (default = None)

    """
    neutral_factor_link = 0

    def __init__(
        self, use_regression_type=True, use_normalization=True, explicit_smoothers=None
    ):
        self.use_regression_type = use_regression_type
        self.use_normalization = use_normalization
        self.explicit_smoothers = self._validate_explicit_smoothers(explicit_smoothers)
        self.onedim_smoothers = _default_smoother_types(
            self.neutral_factor_link, use_normalization
        )

    @staticmethod
    def _validate_explicit_smoothers(explicit_smoothers):
        if explicit_smoothers is None:
            return {}

        def is_tuple_of_strings(x):
            return isinstance(x, tuple) and all(isinstance(s, str) for s in x)

        if not all(
            is_tuple_of_strings(feature_group)
            for feature_group in explicit_smoothers.keys()
        ):
            raise ValueError(
                "All explicit smoothers passed to the SmootherChoice"
                " need to have a tuple of strings as a feature group key."
            )

        if not all(
            isinstance(sm, AbstractBinSmoother) for sm in explicit_smoothers.values()
        ):
            raise ValueError(
                "All explicit smoothers passed to the SmootherChoice"
                " need to be instances of AbstractBinSmoother."
            )

        return explicit_smoothers

    def choice_fct(self, feature_group, feature_property, feature_type=None):
        """
        Returns the smoother specified by the `get_raw_smoother` method
        If an explicit smoother is defined for the feature group,
        the explicit smoother is used instead.

        The result is wrapped with a meta_smoother using the `wrap_smoother` method.

        Parameters
        ----------

        feature_group: tuple
            Name of the feature_group

        feature_property: tuple
            Tuple of feature properties.

        feature_type:
            Type of the feature.
        """

        explicit_smoother = self.explicit_smoothers.get(feature_group)

        if explicit_smoother is not None:
            smoother = explicit_smoother
        else:
            smoother = self.get_raw_smoother(
                feature_group, feature_property, feature_type
            )

        return self.wrap_smoother(
            smoother, feature_group, feature_property, feature_type
        )

    def get_onedim_smoother(self, feature_property, feature_name=None):
        """
        Returns the standard one-dimensional smoother to be used for a
        specific feature.
        If an explicit 1D-smoother for the feature is defined, it is returned,
        otherwise the default smoother for the feature property is chosen.

        Parameters
        ----------

        feature_property: int
            Feature property defined as flag.

        feature_name: str
            Name of the feature

        """
        feature_group = (feature_name,)
        explicit_smoother = self.explicit_smoothers.get(feature_group)

        if explicit_smoother is not None:
            return explicit_smoother
        else:
            return self.onedim_smoothers[_simplify_flags(feature_property)]

    def get_raw_smoother(self, feature_group, feature_property, feature_type=None):
        """Method returning the raw smoother for the `feature_group`,
        `feature_property` and `feature_type` specified.

        This is smoother is not yet wrapped with a `meta_smoother` from the
        `wrap_smoother` method.

        Parameters
        ----------

        feature_group: tuple
            Name of the feature_group

        feature_property: tuple
            Tuple of feature properties.

        feature_type:
            Type of the feature.
        """
        raise NotImplementedError("Please implement this method.")

    def wrap_smoother(self, smoother, feature_group, feature_property, feature_type):
        """Wrapper method that chooses the correct meta_estimators
        for the smoothers in the SmootherChoice class.

        Parameters
        ----------

        smoother: :class:`AbstractBinSmoother`
            smoother that should be wrapped.

        feature_group: tuple
            Name of the feature_group

        feature_property: tuple
            Tuple of feature properties.

        feature_type:
            Type of the feature.
        """
        reg_type = None
        if self.use_regression_type:
            reg_type = determine_reg_type(feature_group, feature_property, feature_type)
        return determine_meta_smoother(smoother, self.use_normalization, reg_type)


class SmootherChoiceWeightedMean(SmootherChoice):
    r"""Weighted mean smoothing for multi-dimensional feature groups.

    This defines a set of common smoothers where
    choose_smoothers_for_factor_model selects from.
    """

    def get_raw_smoother(self, feature_group, feature_prop, feature_type=None):
        if len(feature_group) > 1:
            smoother = smoothing.multidim.WeightedMeanSmoother(
                prior_prediction=self.neutral_factor_link
            )
        else:
            smoother = self.get_onedim_smoother(feature_prop[0], feature_group[0])
        return smoother


class SmootherChoiceGroupBy(SmootherChoice):
    """
    Groupby smoothing for multi-dimensional feature groups.
    """

    def wrap_smoother(
        self, smoother, feature_group, feature_property, feature_type=None
    ):
        # only the properties of the innermost feature should
        # determine the regression type of the groupby smoother:
        if not isinstance(smoother, smoothing.multidim.GroupBySmootherCB):
            return super(self.__class__, self).wrap_smoother(
                smoother, feature_group, feature_property[-1], feature_type
            )
        else:
            return smoother

    def get_raw_smoother(self, feature_group, feature_prop, feature_type=None):
        innermost_smoother = self.get_onedim_smoother(
            feature_prop[-1], feature_group[-1]
        )
        if len(feature_group) > 1:
            return smoothing.multidim.GroupBySmootherCB(
                self.wrap_smoother(innermost_smoother, feature_group, feature_prop),
                n_dim=len(feature_group),
            )
        else:
            return innermost_smoother


class NoSmootherChoice(SmootherChoice):
    """This SmootherChoice class only use BinValueSmoother for all features
    that do no smoothing. It is thought for experimental use only.
    """

    def choice_fct(self, feature_group, feature_prop, feature_type=None):
        if len(feature_group) == 1:
            smoother = smoothing.onedim.BinValuesSmoother()
        else:
            smoother = smoothing.multidim.BinValuesSmoother()
        return self.wrap_smoother(smoother, feature_group, feature_prop, feature_type)


__all__ = [
    "SmootherChoice",
    "SmootherChoiceWeightedMean",
    "SmootherChoiceGroupBy",
]
