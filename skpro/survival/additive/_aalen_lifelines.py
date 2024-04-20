"""Interface adapter to lifelines Aalen additive surival model."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from skpro.survival.adapters.lifelines import _LifelinesAdapter
from skpro.survival.base import BaseSurvReg


class AalenAdditive(_LifelinesAdapter, BaseSurvReg):
    r"""Aalen additive hazards model, from lifelines.

    Direct interface to ``lifelines.fitters.AalenAdditiveFitter``,
    by ``CamDavidsonPilon``.

    This class fits the regression model:

    .. math::  h(t|x)  = b_0(t) + b_1(t) x_1 + ... + b_N(t) x_N

    that is, the hazard rate is a linear function of the covariates
    with time-varying coefficients.
    This implementation assumes non-time-varying covariates.

    Parameters
    ----------
    fit_intercept: bool, optional (default: True)
      If False, do not attach an intercept (column of ones) to the covariate matrix.
      The intercept, :math:`b_0(t)` acts as a baseline hazard.
    alpha: float, optional (default=0.05)
      the level in the confidence intervals around the estimated survival function,
      for computation of ``confidence_intervals_`` fitted parameter.
    coef_penalizer: float, optional (default: 0)
      Attach a L2 penalizer to the size of the coefficients during regression.
      This improves
      stability of the estimates and controls for high correlation between covariates.
      For example, this shrinks the magnitude of :math:`c_{i,t}`.
    smoothing_penalizer: float, optional (default: 0)
      Attach a L2 penalizer to difference between adjacent (over time) coefficients.
      For example, this shrinks the magnitude of :math:`c_{i,t} - c_{i,t+1}`.

    Attributes
    ----------
    cumulative_hazards_ : DataFrame
        The estimated cumulative hazard
    hazards_ : DataFrame
        The estimated hazards
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the cumulative hazard
    durations: array
        The durations provided
    """

    _tags = {"authors": ["CamDavidsonPilon", "rocreguant", "fkiraly"]}
    # CamDavidsonPilon, rocreguant credit for interfaced estimator

    def __init__(
        self,
        fit_intercept=True,
        alpha=0.05,
        coef_penalizer=0.0,
        smoothing_penalizer=0.0,
    ):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coef_penalizer = coef_penalizer
        self.smoothing_penalizer = smoothing_penalizer

        super().__init__()

    def _get_lifelines_class(self):
        """Getter of the lifelines class to be used for the adapter."""
        from lifelines.fitters.aalen_additive_fitter import AalenAdditiveFitter

        return AalenAdditiveFitter

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {}

        params2 = {
            "fit_intercept": False,
            "alpha": 0.1,
            "coef_penalizer": 0.1,
            "smoothing_penalizer": 0.1,
        }

        return [params1, params2]
