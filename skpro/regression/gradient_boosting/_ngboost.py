"""Adapters to ngboost regressors with probabilistic components."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ShreeshaM07"]

import numpy as np
from sklearn.utils import check_random_state

from skpro.regression.base import BaseProbaRegressor


class NGBoostRegressor(BaseProbaRegressor):
    """
    Constructor for all NGBoost models.

    This class implements the methods that are common to all NGBoost models.
    Unless you are implementing a new kind of regression (e.g. interval-censored, etc.),
    you should probably use one of NGBRegressor, NGBClassifier, or NGBSurvival.

    Parameters
    ----------
        dist              : string , default = "Normal"
                            distribution that must be used for
                            probabilistic prediction.
                            Available distribution types
                            1. "Normal"
                            2. "Laplace"
                            3. "LogNormal"
                            4. "Poisson"
                            5. "TDistribution"
        score             : string , default = "LogScore"
                            A score from ngboost.scores for LogScore
                            rule to compare probabilistic
                            predictions P̂ to the observed data y.
        estimator         : default learner/estimator: DecisionTreeRegressor()
                            base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor,
        natural_gradient  : boolean , default = True
                            whether natural gradient must be used or not.
        n_estimators      : int , default = 500
                            the number of boosting iterations to fit
        learning_rate     : float , default = 0.01
                            the learning rate
        minibatch_frac    : float, default = 1.0
                            the percent subsample of rows to
                            use in each boosting iteration
        verbose           : boolean, default=True
                            flag indicating whether output
                            should be printed during fitting
        verbose_eval      : int ,default=100
                            increment (in boosting iterations) at
                            which output should be printed
        tol               : float, default = 1e-4
                            numerical tolerance to be used in optimization
        random_state      : int, RandomState instance or None, optional (default=None)
        validation_fraction: Proportion of training data to
                             set aside as validation data for early stopping.
        early_stopping_rounds: int , default = None , optional
                                The number of consecutive
                                boosting iterations during which the
                                loss has to increase before the algorithm stops early.
                                Set to None to disable early stopping and validation
                                None enables running over the full data set.


    Returns
    -------
        An NGBRegressor object that can be fit.
    """

    _tags = {
        "authors": ["ShreeshaM07"],
        "maintainers": ["ShreeshaM07"],
        "python_dependencies": "ngboost",
    }

    def __init__(
        self,
        dist="Normal",
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
        validation_fraction=0.1,
        early_stopping_rounds=None,
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
        self.random_state = check_random_state(random_state)
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds

        super().__init__()

    def _dist_to_ngboost_instance(self, dist):
        """
        Convert string to NGBoost object.

        self.dist the input string for the type of Distribution.
        It then creates an object of that particular NGBoost Distribution.

        Returns
        -------
        NGBoost Distribution object.
        """
        from ngboost.distns import Laplace, LogNormal, Normal, Poisson, T

        ngboost_dists = {
            "Normal": Normal,
            "Laplace": Laplace,
            "TDistribution": T,
            "Poisson": Poisson,
            "LogNormal": LogNormal,
        }
        # default Normal distribution
        dist_ngboost = Normal
        if dist in ngboost_dists:
            dist_ngboost = ngboost_dists[dist]

        return dist_ngboost

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
        from ngboost import NGBRegressor
        from ngboost.scores import LogScore
        from sklearn.tree import DecisionTreeRegressor

        if self.estimator is None:
            self.estimator = DecisionTreeRegressor(
                criterion="friedman_mse",
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_depth=3,
                splitter="best",
                random_state=None,
            )

        dist_ngboost = self._dist_to_ngboost_instance(self.dist)

        # Score argument for NGBRegressor
        ngboost_score = {
            "LogScore": LogScore,
        }
        score = None
        if self.score in ngboost_score:
            score = ngboost_score[self.score]

        self.ngb_ = NGBRegressor(
            Dist=dist_ngboost,
            Score=score,
            Base=self.estimator,
            natural_gradient=True,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            col_sample=self.col_sample,
            verbose=self.verbose,
            verbose_eval=self.verbose_eval,
            tol=self.tol,
            random_state=self.random_state,
            validation_fraction=self.validation_fraction,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        from sklearn.base import clone

        self.ngb = clone(self.ngb_)
        self.ngb.fit(X, y)
        return self

    def _predict(self, X):
        return self.ngb.predict(X)

    def _pred_dist(self, X):
        return self.ngb.pred_dist(X)

    def _ngb_dist_to_skpro(self, **kwargs):
        """
        Convert NGBoost distribution object to skpro BaseDistribution object.

        Parameters
        ----------
        pred_mean, pred_std and index and columns.

        Returns
        -------
        skpro_dist (skpro.distributions.BaseDistribution):
        Converted skpro distribution object.
        """
        from skpro.distributions.laplace import Laplace
        from skpro.distributions.lognormal import LogNormal
        from skpro.distributions.normal import Normal
        from skpro.distributions.poisson import Poisson
        from skpro.distributions.t import TDistribution

        ngboost_dists = {
            "Normal": Normal,
            "Laplace": Laplace,
            "TDistribution": TDistribution,
            "Poisson": Poisson,
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
        # Normal            | loc = mean, scale = standard deviation
        # TDistribution     | loc = mean, scale = standard deviation
        # Poisson           | mu = mean
        # LogNormal         | s = standard deviation, scale = exp(mean)
        #                   |     (see scipy.stats.lognorm)
        # Laplace           | loc = mean, scale = scale parameter

        dist_params = {
            "Normal": ["loc", "scale"],
            "Laplace": ["loc", "scale"],
            "TDistribution": ["loc", "scale"],
            "Poisson": ["mu"],
            "LogNormal": ["scale", "s"],
        }

        skpro_params = {
            "Normal": ["mu", "sigma"],
            "Laplace": ["mu", "scale"],
            "TDistribution": ["mu", "sigma"],
            "Poisson": ["mu"],
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
            kwargs["columns"] = X.columns

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
        params1 = {"dist": "Normal"}
        params2 = {
            "dist": "Laplace",
            "n_estimators": 800,
        }
        params3 = {}
        params4 = {
            "dist": "Poisson",
            "minibatch_frac": 0.8,
            "early_stopping_rounds": 4,
        }
        params5 = {
            "dist": "LogNormal",
            "learning_rate": 0.001,
            "validation_fraction": 0.2,
        }

        params6 = {
            "dist": "TDistribution",
            "natural_gradient": False,
            "verbose": False,
        }

        return [params1, params2, params3, params4, params5, params6]
