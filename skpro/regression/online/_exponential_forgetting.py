"""Meta-strategy for online learning: exponential forgetting."""

__author__ = ["patelchaitany"]
__all__ = ["OnlineExponentialForgetting"]

import numpy as np
import pandas as pd

from skpro.regression.base import _DelegatedProbaRegressor


class OnlineExponentialForgetting(_DelegatedProbaRegressor):
    """Online regression strategy by exponential forgetting of old data.

    In ``fit``, fits the regressor on all provided data and assigns each
    observation a weight of 1.

    In ``update``, decays the weight of all previously seen observations
    by ``alpha``. New observations enter with weight 1. Observations whose
    weight drops below ``min_weight`` are pruned. The regressor is then
    refitted on the surviving (weighted) data.

    Acts as a soft sliding window where the effective window length is
    approximately ``log(min_weight) / log(alpha)`` updates.

    Caveat: data indices are reset to RangeIndex internally, even if some indices
    passed in ``fit`` and ``update`` overlap.

    Parameters
    ----------
    estimator : skpro regressor, descendant of BaseProbaRegressor
        regressor to be update-refitted on weighted data, blueprint
    alpha : float, default=0.95
        decay factor applied to all existing observation weights at each update.
        Must be in (0, 1). Smaller values forget old data faster.
    min_weight : float, default=1e-3
        minimum weight threshold. Observations with weight below this
        value are pruned before refitting. Must be in (0, 1].

    Attributes
    ----------
    estimator_ : skpro regressor, descendant of BaseProbaRegressor
        clone of the regressor passed in the constructor, fitted on surviving data
    """

    _tags = {"capability:update": True}

    def __init__(self, estimator, alpha=0.95, min_weight=1e-3):
        self.estimator = estimator
        self.alpha = alpha
        self.min_weight = min_weight

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

        # remember data and initialise weights
        self._X = X
        self._y = y
        self._C = C
        self._weights = pd.Series(np.ones(len(X)), index=X.index)

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
        alpha = self.alpha
        min_weight = self.min_weight

        # decay existing weights
        self._weights = self._weights * alpha

        # concatenate new data
        n_new = len(X)

        X_pool = self._update_data(self._X, X)
        y_pool = self._update_data(self._y, y)
        C_pool = self._update_data(self._C, C)

        # append weights of 1 for new observations
        new_weights = pd.Series(np.ones(n_new))
        self._weights = pd.concat([self._weights, new_weights], ignore_index=True)

        # prune observations below min_weight
        mask = self._weights >= min_weight

        X_pool = X_pool.loc[mask.values].reset_index(drop=True)
        y_pool = y_pool.loc[mask.values].reset_index(drop=True)
        if C_pool is not None:
            C_pool = C_pool.loc[mask.values].reset_index(drop=True)
        self._weights = self._weights.loc[mask.values].reset_index(drop=True)

        # user-provided sample_weight via metadata? not supported yet

        # refit on surviving data
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
            {"estimator": regressor, "alpha": 0.9, "min_weight": 1e-2},
        ]

        if _check_estimator_deps(CoxPH, severity="none"):
            coxph = CoxPH()
            params.append({"estimator": coxph})
        else:
            ridge = Ridge()
            params.append({"estimator": ResidualDouble(ridge)})

        return params
