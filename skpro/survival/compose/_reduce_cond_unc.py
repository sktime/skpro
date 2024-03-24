"""Reducer to supervised regression - condition on uncensored."""

__author__ = ["fkiraly"]

import pandas as pd

from skpro.regression.base import BaseProbaRegressor


class ConditionUncensored(BaseProbaRegressor):
    """Reduction to tabular probabilistic regression - conditioning on uncensored.

    Simple baseline reduction strategy for predictive survival analysis.

    Fits a probabilistic regressor on X padded with censoring information C,
    in predict applies the fitted regressor to X padded with 0 (non-censord).

    In ``fit``, passes column concat of ``X`` and ``C`` to ``regressor.fit``.

    In ``predict_[method]``, calls ``regressor.predict_[method]``
    on ``X`` with an additional ``C``-like column, padded with 0,
    and returns the result.

    Parameters
    ----------
    estimator : skpro regressor, BaseProbaRegressor descendant
        probabilistic regressor to predict survival time from features

    Attributes
    ----------
    estimator_ : skpro regressor, BaseProbaRegressor descendant
        fitted probabilistic regressor, clone of ``regressor``
    """

    _tags = {"capability:survival": True}

    def __init__(self, estimator):
        self.estimator = estimator

        super().__init__()

    def _fit(self, X, y, C=None):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Changes state to "fitted" = sets is_fitted flag to True

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pd.DataFrame, must be same length as X
            labels to fit regressor to
        C : pd.DataFrame, optional (default=None)
            censoring information for survival analysis,
            should have same column name as y, same length as X and y
            should have entries 0 and 1 (float or int)
            0 = uncensored, 1 = (right) censored
            if None, all observations are assumed to be uncensored
            Can be passed to any probabilistic regressor,
            but is ignored if capability:survival tag is False.
        """
        self._y_cols = y.columns

        X_and_C = self._get_padded_X(X, C)
        self.estimator_ = self.estimator.clone().fit(X_and_C, y)

        return self

    def _get_padded_X(self, X, C=None):
        """Get one-padded X to use in fit and predict methods."""
        X = X.copy()
        columns = self._y_cols
        index = X.index

        if C is None:
            C = pd.DataFrame(0, index=index, columns=columns)
        else:
            C = C.copy()
        X_and_C = pd.concat([X, C], axis=1)
        return X_and_C

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
            labels predicted for `X`
        """
        X_and_C = self._get_padded_X(X)
        y_pred = self.estimator_.predict(X_and_C)
        return y_pred

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
        X_and_C = self._get_padded_X(X)
        y_pred = self.estimator_.predict_proba(X_and_C)
        return y_pred

    def _predict_interval(self, X, coverage):
        """Compute/return interval predictions.

        private _predict_interval containing the core logic,
            called from predict_interval and default _predict_quantiles

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
            Entries are lower/upper bounds of interval predictions,
            for var in col index, at nominal coverage in second col index,
            lower/upper depending on third col index, for the row index.
            Upper/lower interval end are equivalent to
            quantile predictions at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        X_and_C = self._get_padded_X(X)
        y_pred = self.estimator_.predict_interval(X_and_C, coverage=coverage)
        return y_pred

    def _predict_quantiles(self, X, alpha):
        """Compute/return quantile predictions.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for
        alpha : guaranteed list of float
            A list of probabilities at which quantile predictions are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from ``y`` in fit,
                second level being the values of alpha passed to the function.
            Row index is equal to row index of ``X``.
            Entries are quantile predictions, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        X_and_C = self._get_padded_X(X)
        y_pred = self.estimator_.predict_quantiles(X_and_C, alpha=alpha)
        return y_pred

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
        from skpro.regression.bootstrap import BootstrapRegressor
        from skpro.regression.residual import ResidualDouble

        param1 = {"estimator": ResidualDouble.create_test_instance()}
        param2 = {"estimator": BootstrapRegressor.create_test_instance()}

        return [param1, param2]
