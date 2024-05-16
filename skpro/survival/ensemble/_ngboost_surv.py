"""class for NGBoost probabilistic survival regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ShreeshaM07"]

import numpy as np

from skpro.regression.adapters.ngboost._ngboost_proba import NGBoostAdapter
from skpro.survival.base import BaseSurvReg


class NGBoostSurvival(BaseSurvReg, NGBoostAdapter):
    """Interface of NGBSurvival of ngboost in skpro.

    NGBSurvival is a wrapper for the generic NGBoost class that
    facilitates survival analysis.
    Use this class if you want to predict an outcome that
    could take an infinite number of
    (ordered) values, but right-censoring is present in the observed data.

    Parameters
    ----------
    dist : string , default = "LogNormal"
        assumed distributional form of Y|X=x.
        A distribution from ngboost.distns, e.g. LogNormal
        Available distribution types
        1. "LogNormal"
        2. "Exponential"
    score : string , default = "LogScore"
        rule to compare probabilistic predictions P̂ to the observed data y.
        A score from ngboost.scores, e.g. LogScore
    estimator : default learner/estimator: DecisionTreeRegressor()
        base learner to use in the boosting algorithm.
        Any instantiated sklearn regressor.
    natural_gradient : boolean , default = True
        whether natural gradient must be used or not.
    n_estimators : int , default = 500
        the number of boosting iterations to fit
    learning_rate : float , default = 0.01
        the learning rate
    minibatch_frac : float, default = 1.0
        the percent subsample of rows to
        use in each boosting iteration
    verbose : boolean, default=True
        flag indicating whether output
        should be printed during fitting
    verbose_eval : int ,default=100
        increment (in boosting iterations) at
        which output should be printed
    tol : float, default = 1e-4
        numerical tolerance to be used in optimization
    random_state : int, RandomState instance or None, optional (default=None)

    Returns
    -------
        An NGBSurvival object that can be fit.
    """

    _tags = {
        "authors": ["ShreeshaM07"],
        "maintainers": ["ShreeshaM07"],
        "python_dependencies": "ngboost",
    }

    def __init__(
        self,
        dist="LogNormal",
        score="LogScore",
        estimator=None,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
    ):
        self.dist = dist
        self.score = score
        self.estimator = estimator
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.tol = tol
        self.random_state = random_state

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

        Returns
        -------
        self : reference to self
        """
        import pandas as pd
        from ngboost import NGBSurvival
        from ngboost.scores import LogScore
        from sklearn.tree import DecisionTreeRegressor

        # skpro => 0 = uncensored, 1 = (right) censored
        # ngboost => 1 = uncensored, else (right) censored
        # If C is None then C is set as 1s (uncensored)
        # else it is converted from skpro to ngboost format
        # by doing C = 1-C
        if C is None:
            C = pd.DataFrame(np.ones(len(y)), index=y.index, columns=y.columns)
        else:
            C = 1 - C

        # coerce y to numpy array
        y = self._check_y(y=y)
        y = y[0]
        # remember y columns to predict_proba
        self._y_cols = y.columns
        y = y.values.ravel()

        if self.estimator is None:
            self.estimator_ = DecisionTreeRegressor(
                criterion="friedman_mse",
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_depth=3,
                splitter="best",
                random_state=None,
            )

        dist_ngboost = self._dist_to_ngboost_instance(self.dist, survival=True)

        # Score argument for NGBSurvival
        ngboost_score = {
            "LogScore": LogScore,
        }
        score = None
        if self.score in ngboost_score:
            score = ngboost_score[self.score]

        self.ngbsurv_ = NGBSurvival(
            Dist=dist_ngboost,
            Score=score,
            Base=self.estimator_,
            natural_gradient=True,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            col_sample=self.col_sample,
            verbose=self.verbose,
            verbose_eval=self.verbose_eval,
            tol=self.tol,
            random_state=self.random_state,
        )

        # from sklearn.base import clone

        # self.ngbsurv_ = clone(self.ngbsurv)
        self.ngbsurv_.fit(X, y, C)
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
        import pandas as pd

        df = pd.DataFrame(self.ngbsurv_.predict(X), index=X.index, columns=self._y_cols)
        return df

    def _pred_dist(self, X):
        return self.ngbsurv_.pred_dist(X)

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
        X = self._check_X(X)

        kwargs = {}
        pred_dist = self._pred_dist(X)
        index = X.index
        columns = self._y_cols

        # Convert NGBoost Distribution return params into a dict
        kwargs = self._ngb_skpro_dist_params(pred_dist, index, columns, **kwargs)

        # Convert NGBoost Distribution to skpro BaseDistribution
        pred_dist = self._ngb_dist_to_skpro(**kwargs)

        return pred_dist

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
            "dist": "LogNormal",
            "learning_rate": 0.001,
        }
        params3 = {
            "n_estimators": 800,
            "minibatch_frac": 0.8,
        }
        params4 = {
            "dist": "Exponential",
            "n_estimators": 600,
        }

        return [params1, params2, params3, params4]
