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
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Normal
        Score             : rule to compare probabilistic
                            predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor,
                            e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural
                            gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to
                            use in each boosting iteration
        verbose           : flag indicating whether output
                            should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at
                            which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility.
        validation_fraction: Proportion of training data to
                             set aside as validation data for early stopping.
        early_stopping_rounds: The number of consecutive
            boosting iterations during which the
            loss has to increase before the algorithm stops early.
            Set to None to disable early stopping and validation
            None enables running over the full data set.


    Returns
    -------
        An NGBRegressor object that can be fit.
    """

    _tags = {
        "python_dependencies": "ngboost",
    }

    from ngboost.distns import Laplace, LogNormal, Normal, T
    from ngboost.learners import default_tree_learner
    from ngboost.scores import LogScore

    def __init__(
        self,
        dist="Normal",
        score=LogScore,
        base=default_tree_learner,
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
        self.base = base
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.init_params = None
        self.n_features = None
        self.base_models = []
        self.scalings = []
        self.col_idxs = []
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.best_val_loss_itr = None
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds

        super().__init__()

    def dist_to_ngboost_instance(self):
        """
        Convert string to NGBoost object.

        self.dist the input string for the type of Distribution.
        It then creates an object of that particular NGBoost Distribution.

        Returns
        -------
        NGBoost Distribution object.
        """
        from ngboost.distns import Laplace, LogNormal, Normal, Poisson, T

        dist_ngboost = Normal
        if self.dist == "Normal":
            dist_ngboost = Normal
        elif self.dist == "Laplace":
            dist_ngboost = Laplace
        elif self.dist == "TDistribution":
            dist_ngboost = T
        elif self.dist == "Poisson":
            dist_ngboost = Poisson
        elif self.dist == "LogNormal":
            dist_ngboost = LogNormal

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
        from ngboost.learners import default_tree_learner

        dist_ngboost = self.dist_to_ngboost_instance()

        self.ngb = NGBRegressor(
            Dist=dist_ngboost,
            Score=self.score,
            Base=default_tree_learner,
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
        self.ngb.fit(X, y)
        self._is_fitted = True
        return self

    def _predict(self, X):
        return self.ngb.predict(X)

    def _pred_dist(self, X):
        return self.ngb.pred_dist(X)

    def _ngb_dist_to_skpro(self, pred_mean, pred_std, index, columns):
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
        skpro_dist = None
        if self.dist == "Normal":
            from skpro.distributions.normal import Normal

            skpro_dist = Normal(
                mu=pred_mean, sigma=pred_std, index=index, columns=columns
            )

        if self.dist == "Laplace":
            from skpro.distributions.laplace import Laplace

            skpro_dist = Laplace(
                mu=pred_mean, scale=pred_std, index=index, columns=columns
            )

        if self.dist == "TDistribution":
            from skpro.distributions.t import TDistribution

            skpro_dist = TDistribution(
                mu=pred_mean, sigma=pred_std, index=index, columns=columns
            )

        if self.dist == "Poisson":
            from skpro.distributions.poisson import Poisson

            skpro_dist = Poisson(mu=pred_mean, index=index, columns=columns)

        if self.dist == "LogNormal":
            from skpro.distributions.lognormal import LogNormal

            skpro_dist = LogNormal(
                mu=pred_mean, sigma=pred_std, index=index, columns=columns
            )

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

        # Convert NGBoost distribution to skpro BaseDistribution
        pred_mean = self._predict(X=X)
        if self.dist == "Poisson":
            pred_mean = self._check_y(y=pred_mean)
            # returns a tuple so taking only first index of the tuple
            pred_mean = pred_mean[0]
            pred_std = np.sqrt(pred_mean)
            index = pred_mean.index
            columns = pred_mean.columns
            # converting the ngboost Distribution to a skpro equivalent BaseDistribution
            pred_dist = self._ngb_dist_to_skpro(pred_mean, pred_std, index, columns)
            return pred_dist

        pred_std = np.sqrt(self._pred_dist(X).params["scale"])
        pred_std = self._check_y(y=pred_std)
        # returns a tuple so taking only first index of the tuple
        pred_std = pred_std[0]

        pred_mean = self._check_y(y=pred_mean)
        # returns a tuple so taking only first index of the tuple
        pred_mean = pred_mean[0]

        index = pred_mean.index
        columns = pred_mean.columns

        # converting the ngboost Distribution to an skpro equivalent BaseDistribution
        pred_dist = self._ngb_dist_to_skpro(pred_mean, pred_std, index, columns)

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

        return [params1, params2]


# #Load Boston housing dataset
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# Y = raw_df.values[1::2, 2]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# from sklearn.datasets import load_diabetes
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# from skpro.regression.residual import ResidualDouble

# # step 1: data specification
# X, y = load_diabetes(return_X_y=True, as_frame=True)
# X_train, X_test, Y_train, Y_test = train_test_split(X, y)
# ngb = NGBoostRegressor()._fit(X_train, Y_train)
# Y_preds = ngb._predict(X_test)

# Y_dists = ngb._pred_dist(X_test)

# print(Y_dists)
# Y_pred_proba = ngb.predict_proba(X_test)
# print(Y_pred_proba)

# # test Mean Squared Error
# test_MSE = mean_squared_error(Y_preds, Y_test)
# print('Test MSE', test_MSE)

# # test Negative Log Likelihood
# test_NLL = -Y_dists.logpdf(Y_test).mean()
# print('Test NLL', test_NLL)
