"""MAPIE Cross Conformal Regressor."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Omswastik-11", "vtaquet", "vincentblot28", "TMorzadec", "gmartinonQM"]

import numpy as np
import pandas as pd

from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class MapieCrossConformalRegressor(BaseProbaRegressor):
    """MAPIE Cross Conformal Regressor.

    Direct interface to ``mapie.regression.CrossConformalRegressor`` from the
    ``mapie`` package (>= 1.0).

    Parameters
    ----------
    estimator : sklearn regressor, default=None
        Regressor with scikit-learn compatible API.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    method : str, default="plus"
        Method to choose for prediction interval estimates.
        "plus": Jackknife+
        "minmax": Jackknife-minmax
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        The verbosity level.
    random_state : int, RandomState instance or None, default=None
    """

    _tags = {
        "authors": [
            "Omswastik-11",
            "vtaquet",
            "vincentblot28",
            "TMorzadec",
            "gmartinonQM",
        ],
        "maintainers": ["fkiraly", "Omswastik-11"],
        "python_dependencies": ["MAPIE>=1.0"],
        "capability:missing": True,
    }

    def __init__(
        self,
        estimator=None,
        cv=None,
        method="plus",
        n_jobs=None,
        verbose=0,
        random_state=None,
    ):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        from mapie.regression import CrossConformalRegressor

        X = prep_skl_df(X, copy_df=True)

        self.mapie_est_ = CrossConformalRegressor(
            estimator=self.estimator,
            cv=self.cv,
            method=self.method,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            confidence_level=0.5,  # Dummy
        )

        # Convert y to 1D array for sklearn compatibility
        y_values = y.values.ravel() if hasattr(y, "values") else np.asarray(y).ravel()

        self.mapie_est_.fit_conformalize(X, y_values)
        self._y_cols = y.columns if hasattr(y, "columns") else ["y"]

        return self

    def _predict(self, X):
        X = prep_skl_df(X, copy_df=True)
        y_pred_np = self.mapie_est_.predict(X)
        return pd.DataFrame(y_pred_np, index=X.index, columns=self._y_cols)

    def _predict_interval(self, X, coverage):
        X = prep_skl_df(X, copy_df=True)
        return _predict_interval_mapie(self, X, coverage)

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
            Parameters to create testing instances of the class.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.tree import DecisionTreeRegressor

        params1 = {
            "estimator": LinearRegression(),
        }
        params2 = {
            "estimator": Ridge(alpha=0.5),
            "cv": 3,
            "method": "plus",
            "random_state": 42,
        }
        params3 = {
            "estimator": DecisionTreeRegressor(max_depth=3, random_state=0),
            "cv": 2,
            "method": "minmax",
            "random_state": 0,
        }
        params4 = {
            "estimator": RandomForestRegressor(n_estimators=5, random_state=0),
            "cv": 3,
            "method": "plus",
            "n_jobs": 1,
            "random_state": 0,
        }
        return [params1, params2, params3, params4]


def _predict_interval_mapie(self, X, coverage):
    """Predict intervals with dynamic coverage."""
    results = {}
    original_alphas = self.mapie_est_._alphas.copy()

    try:
        for cov in coverage:
            alpha = 1 - cov
            self.mapie_est_._alphas = [alpha]

            _, intervals = self.mapie_est_.predict_interval(X)

            intervals = np.squeeze(intervals)
            if intervals.ndim == 2:
                lower = intervals[:, 0]
                upper = intervals[:, 1]
            elif intervals.ndim == 3:
                lower = intervals[:, 0, 0]
                upper = intervals[:, 1, 0]
            else:
                lower = intervals[:, 0]
                upper = intervals[:, 1]

            results[(cov, "lower")] = lower
            results[(cov, "upper")] = upper

    finally:
        self.mapie_est_._alphas = original_alphas

    return _build_interval_df(self, X, coverage, results)


def _build_interval_df(self, X, coverage, results):
    """Build the interval DataFrame from results."""
    index = X.index
    columns = pd.MultiIndex.from_product(
        [self._y_cols, coverage, ["lower", "upper"]],
    )

    values = np.zeros((len(X), len(coverage) * 2))
    for i, cov in enumerate(coverage):
        values[:, i * 2] = results[(cov, "lower")]
        values[:, i * 2 + 1] = results[(cov, "upper")]

    return pd.DataFrame(values, index=index, columns=columns)
