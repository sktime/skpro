"""class for NGBoost probabilistic survival regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ShreeshaM07"]

import numpy as np

from skpro.survival.base import BaseSurvReg


class NGBoostSurvival(BaseSurvReg):
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

    def _dist_to_ngboost_instance(self, dist):
        """Convert string to NGBoost object.

        Parameters
        ----------
        dist : string
            the input string for the type of Distribution.
            It then creates an object of that particular NGBoost Distribution.

        Returns
        -------
        NGBoost Distribution object.
        """
        from ngboost.distns import LogNormal

        ngboost_dists = {
            "LogNormal": LogNormal,
        }
        # default Normal distribution
        dist_ngboost = LogNormal
        if dist in ngboost_dists:
            dist_ngboost = ngboost_dists[dist]

        return dist_ngboost

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
            same length as X and y
            should have entries 0 and 1 (float or int)
            1 = uncensored, 0 = (right) censored
            if None, all observations are assumed to be uncensored

        Returns
        -------
        self : reference to self
        """
        import pandas as pd
        from ngboost import NGBSurvival
        from ngboost.scores import LogScore
        from sklearn.tree import DecisionTreeRegressor

        # C is a DataFrame with 1s if C is None
        # uncensored = 1, (right) censored = 0
        if C is None:
            C = pd.DataFrame(np.ones(len(y)), index=y.index, columns=y.columns)

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

        dist_ngboost = self._dist_to_ngboost_instance(self.dist)

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

    def _ngb_dist_to_skpro(self, **kwargs):
        """Convert NGBoost distribution object to skpro BaseDistribution object.

        Parameters
        ----------
        pred_mean, pred_std and index and columns.

        Returns
        -------
        skpro_dist (skpro.distributions.BaseDistribution):
        Converted skpro distribution object.
        """
        from skpro.distributions.lognormal import LogNormal

        ngboost_dists = {
            "LogNormal": LogNormal,
        }

        skpro_dist = None

        if self.dist in ngboost_dists:
            skpro_dist = ngboost_dists[self.dist](**kwargs)

        return skpro_dist

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

        # The returned values of the Distributions from NGBoost
        # are different. So based on that they are split into these
        # categories of loc,scale,mu and s.
        # Distribution type | Parameters
        # ------------------|-----------
        # LogNormal         | s = standard deviation, scale = exp(mean)
        #                   |     (see scipy.stats.lognorm)
        dist_params = {
            "LogNormal": ["scale", "s"],
        }

        skpro_params = {
            "LogNormal": ["mu", "sigma"],
        }

        kwargs = {}

        if self.dist in dist_params and self.dist in skpro_params:
            ngboost_params = dist_params[self.dist]
            skp_params = skpro_params[self.dist]
            for ngboost_param, skp_param in zip(ngboost_params, skp_params):
                kwargs[skp_param] = self._pred_dist(X).params[ngboost_param]
                if self.dist == "LogNormal" and ngboost_param == "scale":
                    kwargs[skp_param] = np.log(self._pred_dist(X).params[ngboost_param])

                kwargs[skp_param] = self._check_y(y=kwargs[skp_param])
                # returns a tuple so taking only first index of the tuple
                kwargs[skp_param] = kwargs[skp_param][0]
            kwargs["index"] = X.index
            kwargs["columns"] = self._y_cols

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

        return [params1, params2, params3]
