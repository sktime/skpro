"""Interface adapter to statsmodels cox proportional hazards models."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import numpy as np

from skpro.survival.base import BaseSurvReg


class CoxPH(BaseSurvReg):
    """Cox proportional hazards model, partial likelihood or elastic net, statsmodels.

    Direct interface to ``statsmodels.duration.hazard_regression.PHReg``

    Implements vanilla partial likelihood minimization (``method="lpl"``),
    and elastic net regularized minimization (``method="elastic_net"``).

    Parameters
    ----------
    method : str, optional (default="lpl"), one of {"lpl", "elastic_net}
        Method used to fit the proportional hazards model.
        "lpl": log partial likelihood minimization, vanilla method.
        Corresponds to statsmodels PHReg.fit method.
        "elastic_net": elastic net regularization.
        Corresponds to statsmodels PHReg.fit_regularized method.
    ties : str, optional (default="breslow"), one of {"breslow", "efron"}
        Method used to deal with ties in the data.
        "breslow": Breslow's method.
        "efron": Efron's method.
    missing : str, optional (default="drop"), one of {"drop"}
        Method used to deal with missing data, statsmodels native.
        "drop": Drops all observations with missing data.
        Note: statsmodels currently supports only "drop" for missing data.
        For imputation, pipeline with sklearn.impute.SimpleImputer,
        or another sklearn composable imputer, via skpro Pipeline (or * dunder)
    strata : pd.Index element or coercible, optional (default=None)
        Strata to use for stratified estimation.
        loc column reference to X, to column to be used as strata (categorical)
        If None, no stratification is used.
        If not None, must be a single valid loc column index of X.
        Iterable is used in loc subsetting of X, i.e., X.loc[:, [strata]].
        Corresponds to statsmodels PHReg.strata parameter.
    alpha : float or iterable of float, optional (default=0.0)
        Used only if method="elastic_net", otherwise ignored.
        Regularization parameter for elastic net regularization.
        If float, regularization parameter is the same for all variables.
        If iterable, must be of same length as number of variables (columns) in X.
        Corresponds to statsmodels PHReg.fit_regularized alpha parameter.

    Attributes
    ----------
    results_: statsmodels PHRegResults instance
        results of the fitted model
    """

    _tags = {
        "capability:missing": False,
        "capability:survival": True,
        "python_dependencies": ["statsmodels"],
    }

    def __init__(
        self,
        method="lpl",
        ties="breslow",
        missing="drop",
        strata=None,
        alpha=0.0,
    ):
        self.method = method
        self.ties = ties
        self.missing = missing
        self.strata = strata
        self.alpha = alpha
        super().__init__()

    def _fit(self, X, y, C=None):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to
        C : pd.DataFrame, optional (default=None)
            censoring information for survival analysis,
            should have same column name as y, same length as X and y
            should have entries 0 and 1 (float or int)
            0 = uncensored, 1 = (right) censored
            if None, all observations are assumed to be uncensored

        Returns
        -------
        self : reference to self
        """
        from statsmodels.duration.hazard_regression import PHReg

        self._y_cols = y.columns

        endog = y.to_numpy().flatten()
        exog = X
        status = C.to_numpy().flatten() if C is not None else None

        params = {
            "ties": self.ties,
            "missing": self.missing,
        }

        if self.strata is not None:
            exog, strata = self._get_strata(exog)
            params["strata"] = strata

        model = PHReg(endog=endog, exog=exog, status=status, **params)
        self.model_ = model

        # fit model
        if self.method == "lpl":
            self.results_ = model.fit()
        else:
            self.results_ = model.fit_regularized(method=self.method, alpha=self.alpha)

        return self

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y_pred : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        from copy import deepcopy

        from skpro.distributions.adapters.statsmodels import empirical_from_rvdf

        # boilerplate code to create correct output index
        index = X.index
        y_cols = self._y_cols  # columns from y in fit, not automatically stored
        columns = y_cols

        # get results from statsmodels
        results = self.results_
        params = results.params
        exog = X.to_numpy()
        n_exog = exog.shape[0]

        # unreported bug in statsmodels PHReg.get_distribution
        # stratum_rows is not set correctly, uses rows from fit
        # we set it manually here
        model = self.model_
        model = deepcopy(model)
        model.surv.stratum_rows = [np.array([ix for ix in range(n_exog)])]

        if self.strata is not None:
            exog, strata = self._get_strata(exog)
            kwargs = {"params": params, "exog": exog, "strata": strata}
        else:
            kwargs = {"params": params, "exog": exog}

        # produce predictions from statsmodels
        # contrary to the documentation, this returns rv_discrete_float
        dist = model.get_distribution(**kwargs)
        dist.xk = dist.xk[:n_exog]
        dist.pk = dist.pk[:n_exog]

        # convert results to skpro BaseDistribution child instance
        y_pred = empirical_from_rvdf(dist=dist, index=index, columns=columns)
        return y_pred

    def _get_strata(self, X):
        """Get strata from X.

        Parameters
        ----------
        X : pandas DataFrame
            feature data frame to get strata from

        Returns
        -------
        X_wo_strata : pandas DataFrame
            X without strata column
        strata : 1D np.ndarray
            strata column from X, coerced to 1D np.ndarray
        """
        strata = X.loc[:, [self.strata]].to_numpy().flatten()
        X_wo_strata = X.drop(columns=[self.strata])
        return X_wo_strata, strata

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
            "method": "elastic_net",
            "alpha": 0.1,
            "ties": "efron",
        }

        return [params1, params2]
