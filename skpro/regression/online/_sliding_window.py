"""Meta-strategy for online learning: sliding window refit."""

__author__ = ["patelchaitany"]
__all__ = ["OnlineSlidingWindow"]

import pandas as pd

from skpro.regression.base import _DelegatedProbaRegressor


class OnlineSlidingWindow(_DelegatedProbaRegressor):
    """Online regression strategy by refitting on a sliding window of data.

    In ``fit``, fits the regressor on all provided data.
    In ``update``, concatenates the new data with previously stored data,
    keeps only the most recent ``window_size`` observations, and refits
    the regressor on the retained window.

    Suited for non-stationary data with abrupt distribution shifts.

    Caveat: data indices are reset to RangeIndex internally, even if some indices
    passed in ``fit`` and ``update`` overlap.

    Parameters
    ----------
    estimator : skpro regressor, descendant of BaseProbaRegressor
        regressor to be update-refitted on the sliding window, blueprint
    window_size : int, default=100
        maximum number of most recent observations to retain

    Attributes
    ----------
    estimator_ : skpro regressor, descendant of BaseProbaRegressor
        clone of the regressor passed in the constructor, fitted on the window
    """

    _tags = {"capability:update": True}

    def __init__(self, estimator, window_size=100):
        self.estimator = estimator
        self.window_size = window_size

        super().__init__()

        tags_to_clone = [
            "capability:missing",
            "capability:survival",
        ]
        self.clone_tags(estimator, tags_to_clone)

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
            Can be passed to any probabilistic regressor,
            but is ignored if capability:survival tag is False.

        Returns
        -------
        self : reference to self
        """
        estimator = self.estimator.clone()
        estimator.fit(X=X, y=y, C=C)
        self.estimator_ = estimator

        # remember data
        self._X = X
        self._y = y
        self._C = C

        return self

    def _update(self, X, y, C=None):
        """Update regressor with new batch of training data.

        State required:
            Requires state to be "fitted".

        Writes to self:
            Updates fitted model attributes ending in "_".

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
            Can be passed to any probabilistic regressor,
            but is ignored if capability:survival tag is False.

        Returns
        -------
        self : reference to self
        """
        window_size = self.window_size

        X_pool = self._update_data(self._X, X)
        y_pool = self._update_data(self._y, y)
        C_pool = self._update_data(self._C, C)

        # keep only the most recent window_size observations
        X_pool = X_pool.tail(window_size).reset_index(drop=True)
        y_pool = y_pool.tail(window_size).reset_index(drop=True)
        if C_pool is not None:
            C_pool = C_pool.tail(window_size).reset_index(drop=True)

        estimator = self.estimator.clone()
        estimator.fit(X=X_pool, y=y_pool, C=C_pool)
        self.estimator_ = estimator

        # remember data
        self._X = X_pool
        self._y = y_pool
        self._C = C_pool

        return self

    def _update_data(self, X, X_new):
        """Update data with new batch of training data.

        Treats X_new as data with new indices, even if some indices overlap with X.

        Parameters
        ----------
        X : pandas DataFrame
        X_new : pandas DataFrame

        Returns
        -------
        X_updated : pandas DataFrame
            concatenated data, with reset index
        """
        if X is None and X_new is None:
            return None
        if X is None and X_new is not None:
            return X_new.reset_index(drop=True)
        if X is not None and X_new is None:
            return X.reset_index(drop=True)
        # else, both X and X_new are not None
        X_updated = pd.concat([X, X_new], ignore_index=True)
        return X_updated

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from skbase.utils.dependencies import _check_estimator_deps
        from sklearn.linear_model import LinearRegression, Ridge

        from skpro.regression.residual import ResidualDouble
        from skpro.survival.coxph import CoxPH

        regressor = ResidualDouble(LinearRegression())

        params = [
            {"estimator": regressor},
            {"estimator": regressor, "window_size": 50},
        ]

        if _check_estimator_deps(CoxPH, severity="none"):
            coxph = CoxPH()
            params.append({"estimator": coxph})
        else:
            ridge = Ridge()
            params.append({"estimator": ResidualDouble(ridge)})

        return params
