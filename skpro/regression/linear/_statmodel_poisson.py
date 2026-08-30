# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df

class StatsModelsPoisson(BaseProbaRegressor):
    """Poisson regression adapter for statsmodels GLM.

    Generalized Linear Model with a Poisson distribution and log-link.
    Provides statistical inference capabilities and probabilistic predictions.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    cov_type : str, default='nonrobust'
        The type of covariance matrix to use for uncertainty estimation.
        Options include 'HC0', 'HC1', 'cluster', etc.
    """

    _tags = {
        "capability:multioutput": False,
        "capability:missing": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        "python_dependencies": ["statsmodels"],
    }

    def __init__(self, fit_intercept=True, cov_type="nonrobust"):
        self.fit_intercept = fit_intercept
        self.cov_type = cov_type
        super().__init__()

    def _fit(self, X, y):
        X_inner = prep_skl_df(X)
        y_inner = prep_skl_df(y)
        self._y_cols = y.columns

        # Statsmodels requires explicit addition of constant for intercept
        if self.fit_intercept:
            X_inner = sm.add_constant(X_inner, has_constant="add")

        # Fit the GLM with Poisson family (default link is log)
        self.model_ = sm.GLM(y_inner, X_inner, family=sm.families.Poisson())
        self.results_ = self.model_.fit(cov_type=self.cov_type)

        # Forward parameters for skpro/sklearn compatibility
        self.params_ = self.results_.params
        if self.fit_intercept:
            self.intercept_ = self.params_[0]
            self.coef_ = self.params_[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.params_

        return self

    def _predict(self, X):
        X_inner = prep_skl_df(X)

        if self.fit_intercept:
            X_inner = sm.add_constant(X_inner, has_constant="add")

        # Statsmodels predict returns the mean (mu)
        y_pred = self.results_.predict(X_inner)
        return pd.DataFrame(y_pred, index=X.index, columns=self._y_cols)

    def _predict_var(self, X):
        """Poisson variance is equal to the mean."""
        return self._predict(X)

    def _predict_proba(self, X):
        from skpro.distributions.poisson import Poisson

        # Extract predicted rate (mu/lambda)
        y_pred = self._predict(X).values
        # Return skpro Distribution object
        return Poisson(mu=y_pred, index=X.index, columns=self._y_cols)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"fit_intercept": True}, {"fit_intercept": False, "cov_type": "HC0"}]
