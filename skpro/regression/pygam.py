"""Interface adapter for pygam GAM regressor."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["neha222222"]

import numpy as np
import pandas as pd

from skpro.regression.base import BaseProbaRegressor


class GAMRegressor(BaseProbaRegressor):
    """Generalized Additive Model probabilistic regressor.

    Direct interface to ``pygam.GAM`` from the ``pygam`` package.

    GAM models predict using a sum of smooth functions of features,
    with prediction intervals based on the model's uncertainty estimates.

    Parameters
    ----------
    distribution : str, default="normal"
        Distribution family for the response variable.
        Options: "normal", "poisson", "gamma", "binomial", "inv_gauss".
    link : str, default="identity"
        Link function for the GAM.
        Options depend on distribution: "identity", "log", "logit", "inverse".
    n_splines : int, default=25
        Number of splines to use for each feature.
    spline_order : int, default=3
        Order of spline to use. 3 = cubic splines.
    lam : float or array-like, default=0.6
        Regularization strength. Can be a single value or array for each term.
    max_iter : int, default=100
        Maximum number of iterations for fitting.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    Attributes
    ----------
    gam_ : pygam.GAM
        Fitted GAM model.

    Examples
    --------
    >>> from skpro.regression.pygam import GAMRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg_proba = GAMRegressor()
    >>> reg_proba.fit(X_train, y_train)
    GAMRegressor(...)
    >>> y_pred = reg_proba.predict(X_test)
    >>> y_pred_int = reg_proba.predict_interval(X_test)
    """

    _tags = {
        "authors": ["neha222222"],
        "python_dependencies": ["pygam"],
        "capability:missing": False,
    }

    def __init__(
        self,
        distribution="normal",
        link="identity",
        n_splines=25,
        spline_order=3,
        lam=0.6,
        max_iter=100,
        tol=1e-4,
        fit_intercept=True,
    ):
        self.distribution = distribution
        self.link = link
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

        super().__init__()

    def _fit(self, X, y):
        """Fit regressor to training data.

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
        from pygam import GAM, s

        self._y_cols = y.columns

        X_np = X.to_numpy()
        y_np = y.to_numpy().flatten()

        n_features = X_np.shape[1]
        terms = s(0, n_splines=self.n_splines, spline_order=self.spline_order)
        for i in range(1, n_features):
            terms += s(i, n_splines=self.n_splines, spline_order=self.spline_order)

        gam = GAM(
            terms,
            distribution=self.distribution,
            link=self.link,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            lam=self.lam,
        )

        gam.fit(X_np, y_np)
        self.gam_ = gam

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            labels predicted for `X`
        """
        X_np = X.to_numpy()
        y_pred_np = self.gam_.predict(X_np)

        index = X.index
        columns = self._y_cols
        y_pred = pd.DataFrame(y_pred_np, index=index, columns=columns)

        return y_pred

    def _predict_interval(self, X, coverage):
        """Compute/return interval predictions.

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for
        coverage : guaranteed list of float of unique values
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from ``y`` in fit,
            second level coverage fractions for which intervals were computed,
            in the same order as in input `coverage`.
            Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is equal to row index of ``X``.
        """
        X_np = X.to_numpy()

        all_lower = []
        all_upper = []

        for cov in coverage:
            intervals = self.gam_.confidence_intervals(X_np, width=cov)
            all_lower.append(intervals[:, 0])
            all_upper.append(intervals[:, 1])

        index = X.index
        columns = pd.MultiIndex.from_product(
            [self._y_cols, coverage, ["lower", "upper"]],
        )

        n_samples = len(index)
        n_coverages = len(coverage)
        values = np.zeros((n_samples, n_coverages * 2))

        for i, cov in enumerate(coverage):
            values[:, i * 2] = all_lower[i]
            values[:, i * 2 + 1] = all_upper[i]

        pred_int = pd.DataFrame(values, index=index, columns=columns)

        return pred_int

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
            Parameters to create testing instances of the class
        """
        params1 = {"n_splines": 10, "max_iter": 50}
        params2 = {"distribution": "normal", "n_splines": 5, "lam": 1.0}

        return [params1, params2]

