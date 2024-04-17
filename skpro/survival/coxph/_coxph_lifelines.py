"""Interface adapter to lifelines Cox PH surival model."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from skpro.survival.adapters.lifelines import _LifelinesAdapter
from skpro.survival.base import BaseSurvReg


class CoxPHlifelines(_LifelinesAdapter, BaseSurvReg):
    r"""Cox proportional hazards models, from lifelines.

    Direct interface to ``lifelines.fitters.CoxPHFitter``,
    by ``CamDavidsonPilon``.

    This class implements Cox proportional hazard model,

    .. math::  h(t|x) = h_0(t) \exp((x - \overline{x})' \beta)

    with different options to fit the baseline hazard, :math:`h_0(t)`.

    The class offers multiple options via the ``baseline_estimation_method`` parameter:

    ``"breslow"`` (default): non-parametric estimate via Breslow's method.
    In this case, the entire model is the traditional semi-parametric Cox model.
    Ties are handled using Efron's method.

    ``"spline"``: parametric spline fit of baseline hazard,
    via Royston-Parmar's method [1]_. The parametric form is

    .. math:: H_0(t) = \exp{\left( \phi_0 + \phi_1\log{t} + \sum_{j=2}^N \phi_j v_j(\log{t})\right)}  # noqa E501

    where :math:`v_j` are our cubic basis functions at predetermined knots,
    and :math:`H_0` is the cumulative baseline hazard. See [1]_ for exact definition.

    ``"piecewise"``: non-parametric, piecewise constant empirical baseline hazard.
    The explicit form of the baseline hazard is

    .. math::  h_0(t) = \begin{cases}
        exp{\beta \cdot \text{center}(x)}  & \text{if $t \le \tau_0$} \\
        exp{\beta \cdot \text{center}(x)} \cdot lambda_1 & \text{if $\tau_0 < t \le \tau_1$} \\  # noqa E501
        exp{\beta \cdot \text{center}(x)} \cdot lambda_2 & \text{if $\tau_1 < t \le \tau_2$} \\  # noqa E501
        ...
        \end{cases}

    Parameters
    ----------
    alpha: float, optional (default=0.05)
      the level in the confidence intervals around the estimated survival function,
      for computation of ``confidence_intervals_`` fitted parameter.

    baseline_estimation_method: string, default="breslow",
        one of: ``"breslow"``, ``"spline"``, or ``"piecewise"``.
        Specifies algorithm for estimation of baseline hazard, see above.
        If ``"piecewise"``, the ``breakpoints`` parameter must be set.

    penalizer: float or array, optional (default=0.0)
        Penalty to the size of the coefficients during regression.
        This improves stability of the estimates and controls for high correlation
        between covariates.
        For example, this shrinks the magnitude value of :math:`\beta_i`.
        See ``l1_ratio`` below.
        The penalty term is :math:`\text{penalizer} \left( \frac{1-\text{l1_ratio}}{2} ||\beta||_2^2 + \text{l1_ratio}||\beta||_1\right)`.  # noqa E501

        If an array, must be equal in size to the number of parameters,
        with penalty coefficients for specific variables. For
        example, ``penalizer=0.01 * np.ones(p)`` is the same as ``penalizer=0.01``.

    l1_ratio: float, optional (default=0.0)
        Specify what ratio to assign to a L1 vs L2 penalty.
        Same as in scikit-learn. See ``penalizer`` above.

    strata: list, optional
        specify a list of columns to use in stratification. This is useful if a
        categorical covariate does not obey the proportional hazard assumption. This
        is used similar to the ``strata`` expression in R.
        See http://courses.washington.edu/b515/l17.pdf.

    n_baseline_knots: int, optional, default=4
        Used only when ``baseline_estimation_method="spline"``.
        Set the number of knots (interior & exterior) in the baseline hazard,
        which will be placed evenly along the time axis.
        Should be at least 2.
        Royston et. al, the authors of this model, suggest 4 to start,
        but any values between 2 and 8 are reasonable.
        If you need to customize the timestamps used to calculate the curve,
        use the ``knots`` parameter instead.

    knots: list, optional
        Used only when ``baseline_estimation_method="spline"``.
        Specifies custom points in the time axis for the baseline hazard curve.
        To use evenly-spaced points in time, the ``n_baseline_knots``
        parameter can be employed instead.

    breakpoints: list, optional
        Used only when ``baseline_estimation_method="piecewise"``,
        must be passed in this case.
        Set the positions of the baseline hazard breakpoints.

    Attributes
    ----------
    params_ : Series
        The estimated coefficients.
    hazard_ratios_ : Series
        The exp(coefficients)
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the hazard coefficients
    durations: Series
        The durations provided
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : DataFrame
        The variance matrix of the coefficients
    strata: list
        the strata provided
    standard_errors_: Series
        the standard errors of the estimates
    log_likelihood_: float
        the log-likelihood at the fitted coefficients
    AIC_: float
        the AIC at the fitted coefficients (if using splines for baseline hazard)
    partial_AIC_: float
        the AIC at the fitted coefficients
        (if using non-parametric inference for baseline hazard)
    baseline_hazard_: DataFrame
        the baseline hazard evaluated at the observed times.
        Estimated using Breslow's method.
    baseline_cumulative_hazard_: DataFrame
        the baseline cumulative hazard evaluated at the observed times.
        Estimated using Breslow's method.
    baseline_survival_: DataFrame
        the baseline survival evaluated at the observed times.
        Estimated using Breslow's method.
    summary: Dataframe
        a Dataframe of the coefficients, p-values, CIs, etc.

    References
    ----------
    .. [1] Royston, P., Parmar, M. K. B. (2002).
      Flexible parametric proportional-hazards and proportional-odds
      models for censored survival data, with application to prognostic
      modelling and estimation of treatment effects.
      Statistics in Medicine, 21(15), 2175–2197. doi:10.1002/sim.1203
    """

    _tags = {"authors": ["CamDavidsonPilon", "JoseLlanes", "mathurinm", "fkiraly"]}
    # CamDavidsonPilon, JoseLlanes, mathurinm credit for interfaced estimator

    def __init__(
        self,
        baseline_estimation_method: str = "breslow",
        penalizer=0.0,
        strata=None,
        l1_ratio=0.0,
        n_baseline_knots=None,
        knots=None,
        breakpoints=None,
    ):
        self.baseline_estimation_method = baseline_estimation_method
        self.penalizer = penalizer
        self.strata = strata
        self.l1_ratio = l1_ratio
        self.n_baseline_knots = n_baseline_knots
        self.knots = knots
        self.breakpoints = breakpoints

        super().__init__()

    def _get_lifelines_class(self):
        """Getter of the lifelines class to be used for the adapter."""
        from lifelines.fitters.coxph_fitter import CoxPHFitter

        return CoxPHFitter

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

        # breakpoints are specific to data ranges,
        # but tests loop over various data sets, so this would break
        #
        # params3 = {
        #     "baseline_estimation_method": "piecewise",
        #     "penalizer": 0.15,
        #     "l1_ratio": 0.05,
        #     "breakpoints": [10, 20, 30, 100],
        # }

        return [params1, params2]
