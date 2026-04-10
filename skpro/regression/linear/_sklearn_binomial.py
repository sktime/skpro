"""Adapters to sklearn linear regressors with probabilistic components."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# based on sktime pipelines

import pandas as pd

from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class BinomialRegressor(BaseProbaRegressor):
    """Binomial regression, direct adapter to sklearn LogisticRegression.

    Generalized Linear Model with a Binomial distribution.
    This regressor uses the 'logit' link function.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'}, default='lbfgs'
        Algorithm to use in the optimization problem.

    max_iter : int, default=100
        The maximal number of iterations for the solver.

    tol : float, default=1e-4
        The convergence tolerance.

    verbose : int, default=0
        For the 'lbfgs', 'sag' and 'saga' solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model.

    intercept_ : float
        Independent term in decision function.

    n_iter_ : int
        The actual number of iterations before reaching the stopping criterion.

    n_features_in_ : int
        Number of features seen during :term:'fit'.

    feature_names_in_ : ndarray of shape (n_features,)
        Names of features seen during :term:'fit'.
    """

    _tags = {
        "capability:multioutput": False,
        "capability:missing": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(
        self,
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        verbose=0,
        warm_start=False,
        solver="lbfgs",
    ):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.solver = solver

        super().__init__()

        from sklearn.linear_model import LogisticRegression

        skl_estimator = LogisticRegression(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            solver=solver,
        )

        self.estimator_ = skl_estimator

    FITTED_PARAMS_TO_FORWARD = [
        "coef_",
        "intercept_",
        "n_iter_",
    ]

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        X_inner = prep_skl_df(X).to_numpy()
        y_inner = prep_skl_df(y).to_numpy()

        self._y_cols = y.columns

        if len(y_inner.shape) > 1 and y_inner.shape[1] == 1:
            y_inner = y_inner[:, 0]

        estimator = self.estimator_
        estimator.fit(X=X_inner, y=y_inner)

        for attr in self.FITTED_PARAMS_TO_FORWARD:
            setattr(self, attr, getattr(estimator, attr))

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted" = self.is_fitted=True

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            predicted probabilities for `X`
        """
        X_inner = prep_skl_df(X).to_numpy()
        # probability of class 1
        y_pred = self.estimator_.predict_proba(X_inner)[:, 1]
        y_pred_df = pd.DataFrame(y_pred, index=X.index, columns=self._y_cols)

        return y_pred_df

    def _predict_var(self, X):
        """Compute/return variance predictions."""
        p = self._predict(X)
        return p * (1 - p)  # Binomial variance for n=1

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
        from skpro.distributions.binomial import Binomial

        y_cols = self._y_cols
        p = self.predict(X).values

        y_pred_proba = Binomial(n=1, p=p, index=X.index, columns=y_cols)

        return y_pred_proba

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
        """
        param1 = {}

        param2 = {
            "fit_intercept": False,
            "max_iter": 200,
            "tol": 2e-4,
            "verbose": 1,
            "warm_start": True,
            "solver": "liblinear",
        }

        return [param1, param2]
