"""Adapters to ngboost regressors with probabilistic components."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ShreeshaM07"]

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.manifold import manifold
from ngboost.scores import LogScore
from sklearn.utils import check_random_state

from skpro.regression.base.adapters import _DelegateWithFittedParamForwarding


class NGBoostRegressor(_DelegateWithFittedParamForwarding):
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

    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        Base=default_tree_learner,
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
        self.Dist = Dist
        self.Score = Score
        self.Base = Base
        self.Manifold = manifold(Score, Dist)
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
        self.ngb = NGBRegressor(
            Dist=self.Dist,
            Score=self.Score,
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
        return self

    def _predict(self, X):
        return self.ngb.predict(X)

    def _pred_dist(self, X):
        return self.ngb.pred_dist(X)


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# #Load Boston housing dataset
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# Y = raw_df.values[1::2, 2]

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# ngb = NGBoostRegressor()._fit(X_train, Y_train)
# Y_preds = ngb._predict(X_test)
# Y_dists = ngb._pred_dist(X_test)

# # test Mean Squared Error
# test_MSE = mean_squared_error(Y_preds, Y_test)
# print('Test MSE', test_MSE)

# # test Negative Log Likelihood
# test_NLL = -Y_dists.logpdf(Y_test).mean()
# print('Test NLL', test_NLL)
