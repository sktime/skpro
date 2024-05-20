"""Adapters to ngboost regressors with probabilistic components."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ShreeshaM07"]

from skpro.regression.adapters.ngboost._ngboost_proba import NGBoostAdapter
from skpro.regression.base import BaseProbaRegressor


class NGBoostRegressor(BaseProbaRegressor, NGBoostAdapter):
    """Natural Gradient Boosting Regressor for probabilistic regressors.

    It is an interface to the NGBRegressor.
    NGBRegressor is a wrapper for the generic NGBoost class that facilitates regression.
    Use this class if you want to predict an outcome that could take an
    infinite number of (ordered) values.

    Parameters
    ----------
    dist : string , default = "Normal"
        distribution that must be used for
        probabilistic prediction.
        Available distribution types
        1. "Normal"
        2. "Laplace"
        3. "LogNormal"
        4. "Poisson"
        5. "TDistribution"
        6. "Exponential"
    score : string , default = "LogScore"
        A score from ngboost.scores for LogScore
        rule to compare probabilistic
        predictions P̂ to the observed data y.
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
    validation_fraction : Proportion of training data to
        set aside as validation data for early stopping.
    early_stopping_rounds : int , default = None , optional
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
        self.random_state = random_state
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
        from ngboost import NGBRegressor
        from ngboost.scores import LogScore
        from sklearn.tree import DecisionTreeRegressor

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

        dist_ngboost = self._dist_to_ngboost_instance(self.dist, survival=False)

        # Score argument for NGBRegressor
        ngboost_score = {
            "LogScore": LogScore,
        }
        score = None
        if self.score in ngboost_score:
            score = ngboost_score[self.score]

        self.ngb = NGBRegressor(
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
            validation_fraction=self.validation_fraction,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        from sklearn.base import clone

        self.ngb_ = clone(self.ngb)
        self.ngb_.fit(X, y)
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

        df = pd.DataFrame(self.ngb_.predict(X), index=X.index, columns=self._y_cols)
        return df

    def _pred_dist(self, X):
        return self.ngb_.pred_dist(X)

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
            "dist": "Normal",
            "natural_gradient": False,
            "verbose": False,
        }

        params7 = {
            "dist": "Exponential",
            "n_estimators": 800,
            "verbose_eval": 50,
        }

        return [params1, params2, params3, params4, params5, params6, params7]
