"""Interface adapter to lifelines Weibull AFT model."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from skpro.survival.adapters.lifelines import _LifelinesAdapter
from skpro.survival.base import BaseSurvReg


class AFTWeibullLifelines(_LifelinesAdapter, BaseSurvReg):
    r"""Weibull AFT model, from lifelines.

    Direct interface to ``lifelines.fitters.WeibullAFTFitter``,
    by ``CamDavidsonPilon``.

    This class implements a Weibull AFT model. The model has parametric form, with
    :math:`\lambda(x) = \exp\left(\beta_0 + \beta_1x_1 + ... + \beta_n x_n \right)`,
    and optionally,
    :math:`\rho(y) = \exp\left(\alpha_0 + \alpha_1 y_1 + ... + \alpha_m y_m \right)`,

    with predictive distribution being Weibull, with
    scale parameter :math:`\lambda(x)` and shape parameter (exponent) :math:`\rho(y)`.

    The :math:`\lambda` (scale) parameter is a decay or half-like like parameter,
    more specifically, the time by which the survival probability is 37%.
    The :math:`\rho` (shape) parameter controls curvature of the the cumulative hazard,
    e.g., whether it is convex or concave, representing accelerating or decelerating
    hazards.

    The cumulative hazard rate is

    .. math:: H(t; x, y) = \left(\frac{t}{\lambda(x)} \right)^{\rho(y)},

    Parameters
    ----------
    alpha: float, optional (default=0.05)
      the level in the confidence intervals around the estimated survival function,
      for computation of ``confidence_intervals_`` fitted parameter.

    fit_intercept: boolean, optional (default=True)
        Whether to fit an intercept term in the model.

    penalizer: float or array, optional (default=0.0)
        the penalizer coefficient to the size of the coefficients.
        See ``l1_ratio``. Must be equal to or greater than 0.
        Alternatively, penalizer is an array equal in size to the number of parameters,
        with penalty coefficients for specific variables. For
        example, ``penalizer=0.01 * np.ones(p)`` is the same as ``penalizer=0.01``

    l1_ratio: float, optional (default=0.0)
        how much of the penalizer should be attributed to an l1 penalty
        (otherwise an l2 penalty). The penalty function looks like
        ``penalizer * l1_ratio * ||w||_1 + 0.5 * penalizer * (1 - l1_ratio) * ||w||^2_2``  # noqa E501

    Attributes
    ----------
    params_ : DataFrame
        The estimated coefficients
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the coefficients
    durations: Series
        The event_observed variable provided
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : DataFrame
        The variance matrix of the coefficients
    standard_errors_: Series
        the standard errors of the estimates
    score_: float
        the concordance index of the model.
    """

    _tags = {"authors": ["CamDavidsonPilon", "JoseLlanes", "mathurinm", "fkiraly"]}
    # CamDavidsonPilon, JoseLlanes, mathurinm credit for interfaced estimator

    def __init__(
        self,
        alpha: float = 0.05,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
    ):
        self.alpha = alpha
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept

        super().__init__()

    def _get_lifelines_class(self):
        """Getter of the lifelines class to be used for the adapter."""
        from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter

        return WeibullAFTFitter

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
            "baseline_estimation_method": "spline",
            "penalizer": 0.1,
            "l1_ratio": 0.1,
            "n_baseline_knots": 3,
        }


        return [params1, params2]
