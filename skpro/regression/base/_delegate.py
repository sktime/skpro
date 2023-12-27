"""Delegator mixin that delegates all methods to wrapped regressor.

Useful for building estimators where all but one or a few methods are delegated. For
that purpose, inherit from this estimator and then override only the methods that
are not delegated.
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["_DelegatedProbaRegressor"]

from skpro.regression.base._base import BaseProbaRegressor


class _DelegatedProbaRegressor(BaseProbaRegressor):
    """Delegator mixin that delegates all methods to wrapped regressor.

    Delegates inner methods to a wrapped estimator.
        Wrapped estimator is value of attribute with name self._delegate_name.
        By default, this is "estimator_", i.e., delegates to self.estimator_
        To override delegation, override _delegate_name attribute in child class.

    Delegates the following inner underscore methods:
        _fit, _predict,
        _predict_interval, _predict_quantiles, _predict_var, _predict_proba

    Does NOT delegate get_params, set_params.
        get_params, set_params will hence use one additional nesting level by default.

    Does NOT delegate or copy tags, this should be done in a child class if required.
    """

    # attribute for _DelegatedProbaRegressor, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedRegressor docstring
    _delegate_name = "estimator_"

    def _get_delegate(self):
        return getattr(self, self._delegate_name)

    def _fit(self, X, y, C=None):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

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

        Returns
        -------
        self : reference to self
        """
        estimator = self._get_delegate()
        estimator.fit(X=X, y=y, C=C)
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
            labels predicted for `X`
        """
        estimator = self._get_delegate()
        return estimator.predict(X=X)

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
        estimator = self._get_delegate()
        return estimator.predict_quantiles(X=X, alpha=alpha)

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
        estimator = self._get_delegate()
        return estimator.predict_interval(X=X, coverage=coverage)

    def _predict_var(self, X):
        """Compute/return variance predictions.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        pred_var : pd.DataFrame
            Column names are exactly those of ``y`` passed in ``fit``.
            Row index is equal to row index of ``X``.
            Entries are variance prediction, for var in col index.
            A variance prediction for given variable and fh index is a predicted
            variance for that variable and index, given observed data.
        """
        estimator = self._get_delegate()
        return estimator.predict_var(X=X)

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
        y : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        estimator = self._get_delegate()
        return estimator.predict_proba(X=X)

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        estimator = self._get_delegate()
        return estimator.get_fitted_params()
