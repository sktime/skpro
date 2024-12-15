"""Meta-strategy for online learning: refit on full data."""

__author__ = ["fkiraly"]
__all__ = ["OnlineRefitEveryN"]

import pandas as pd

from skpro.regression.base import _DelegatedProbaRegressor


class OnlineRefitEveryN(_DelegatedProbaRegressor):
    """Online regression strategy, updates only after N new data points are seen.

    In ``fit``, behaves like the wrapped regressor.

    In ``update``, runs the wrapped regressor's ``update`` only if
    at least N new data points have been seen since the last
    ``fit`` or ``update``.

    Can be combined with ``OnlineRefit`` to refit on the entire data,
    but only after N new data points have been seen.

    Parameters
    ----------
    estimator : skpro regressor, descendant of BaseProbaRegressor
        regressor to be update-refitted on all data, blueprint
    N : int, default=1
        number of new data points to see before updating the regressor

    Attributes
    ----------
    estimator_ : skpro regressor, descendant of BaseProbaRegressor
        clone of the regressor passed in the constructor, fitted on all data
    """

    _tags = {"capability:update": True}

    def __init__(self, estimator, N=1):
        self.estimator = estimator
        self.N = N

        super().__init__()

        tags_to_clone = [
            "capability:missing",
            "capability:survival",
        ]
        self.clone_tags(estimator, tags_to_clone)

        self.estimator_ = estimator.clone()

        self._X = None
        self._y = None
        self._C = None
        self.n_seen_since_last_update_ = 0

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
        X_pool = self._update_data(self._X, X)
        y_pool = self._update_data(self._y, y)
        C_pool = self._update_data(self._C, C)

        n_seen_now = len(X)
        n_seen_since_last_update = self.n_seen_since_last_update_ + n_seen_now

        if n_seen_since_last_update >= self.N:
            self.estimator_.update(X=X_pool, y=y_pool, C=C_pool)
            self._X = None
            self._y = None
            self._C = None
            self.n_seen_since_last_update_ = 0
        else:
            self.n_seen_since_last_update_ = n_seen_since_last_update

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
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from skbase.utils.dependencies import _check_estimator_deps
        from sklearn.linear_model import LinearRegression, Ridge

        from skpro.regression.residual import ResidualDouble
        from skpro.survival.coxph import CoxPH

        regressor = ResidualDouble(LinearRegression())

        params = [{"estimator": regressor}]

        if _check_estimator_deps(CoxPH, severity="none"):
            coxph = CoxPH()
            params.append({"estimator": coxph})
        else:
            ridge = Ridge()
            params.append({"estimator": ResidualDouble(ridge)})

        return params
