"""Interface adapter to lifelines Log-Normal AFT model."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import numpy as np

from skpro.distributions.lognormal import LogNormal
from skpro.survival.adapters.lifelines import _LifelinesAdapter
from skpro.survival.base import BaseSurvReg


class AFTLogNormal(_LifelinesAdapter, BaseSurvReg):
    r"""Log-Normal AFT model, from lifelines.

    Direct interface to ``lifelines.fitters.LogNormalAFTFitter``,
    by ``CamDavidsonPilon``.

    This class implements a Log-Normal AFT model. The model has parametric form, with
    :math:`\mu(x) = \exp\left(\beta_0 + \beta_1x_1 + ... + \beta_n x_n \right)`,
    and, optionally,
    :math:`\sigma(y) = \exp\left(\alpha_0 + \alpha_1 y_1 + ... + \alpha_m y_m \right)`,

    with predictive distribution being Log-Normal, with
    mean :math:`\mu(x)` and standard deviation :math:`\sigma(y)`.

    Parameters
    ----------
    mu_cols: pd.Index or coercible, optional, default=None
        Columns of the input data frame to be used as covariates for
        the mean parameter :math:`\mu`.
        If None, all columns are used.

    sd_cols: string "all", pd.Index or coercible, optional, default=None
        Columns of the input data frame to be used as covariates for
        the standard deviation parameter :math:`\sigma`.
        If None, no covariates are used, the standard deviation parameter
        is estimated as a constant. If "all", all columns are used.

    fit_intercept: boolean, optional (default=True)
        Whether to fit an intercept term in the model.

    alpha: float, optional (default=0.05)
      the level in the confidence intervals around the estimated survival function,
      for computation of ``confidence_intervals_`` fitted parameter.

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

    _tags = {"authors": ["CamDavidsonPilon", "fkiraly"]}
    # CamDavidsonPilon, credit for interfaced estimator

    def __init__(
        self,
        mu_cols=None,
        sd_cols=None,
        fit_intercept: bool = True,
        alpha: float = 0.05,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
    ):
        self.mu_cols = mu_cols
        self.sd_cols = sd_cols
        self.alpha = alpha
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept

        super().__init__()

        if mu_cols is not None:
            self.X_col_subset = mu_cols

    def _get_lifelines_class(self):
        """Getter of the lifelines class to be used for the adapter."""
        from lifelines.fitters.log_normal_aft_fitter import LogNormalAFTFitter

        return LogNormalAFTFitter

    def _get_lifelines_object(self):
        """Abstract method to initialize lifelines object.

        The default initializes result of _get_lifelines_class
        with self.get_params.
        """
        cls = self._get_lifelines_class()
        params = self.get_params()
        params.pop("mu_cols", None)
        params.pop("sd_cols", None)
        return cls(**params)

    def _add_extra_fit_args(self, X, y, C=None):
        """Get extra arguments for the fit method.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y: pd.DataFrame
            Training labels
        C: pd.DataFrame, optional (default=None)
            Censoring information for survival analysis.
        fit_args: dict, optional (default=None)
            Existing arguments for the fit method, from the adapter.

        Returns
        -------
        dict
            Extra arguments for the fit method.
        """
        if self.mu_cols is not None:
            if self.mu_cols == "all":
                return {"ancillary": True}
            else:
                return {"ancillary": X[self.mu_cols]}
        else:
            return {}

    def _predict_proba(self, X):
        """Predict_proba method adapter.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on.

        Returns
        -------
        skpro Empirical distribution
        """
        if self.sd_cols == "all":
            ancillary = X
        elif self.sd_cols is not None:
            ancillary = X[self.sd_cols]
        else:
            ancillary = None

        if self.mu_cols is not None:
            df = X[self.mu_cols]
        else:
            df = X

        lifelines_est = getattr(self, self._estimator_attr)
        ll_pred_proba = lifelines_est._prep_inputs_for_prediction_and_return_scores

        mu, sigma = ll_pred_proba(df, ancillary)
        mu = np.expand_dims(mu, axis=1)
        sigma = np.expand_dims(sigma, axis=1)

        dist = LogNormal(mu=mu, sigma=sigma, index=X.index, columns=self._y_cols)
        return dist

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
            "sd_cols": "all",
            "fit_intercept": False,
            "alpha": 0.1,
            "penalizer": 0.001,
            "l1_ratio": 0.001,
        }
        return [params1, params2]
