"""Bagging probabilistic regressors."""

__author__ = ["fkiraly"]
__all__ = ["BaggingRegressor"]

from math import ceil

import numpy as np
import pandas as pd

from skpro.distributions.mixture import Mixture
from skpro.regression.base import BaseProbaRegressor


class BaggingRegressor(BaseProbaRegressor):
    """Bagging ensemble of probabilistic regresesors.

    Fits ``n_estimators`` clones of an skpro regressor on
    datasets which are instance sub-samples and/or variable sub-samples.

    On ``predict_proba``, the mixture of probabilistic predictions is returned.

    The estimator allows to choose sample sizes for instances, variables,
    and whether sampling is with or without replacement.

    Direct generalization of ``sklearn``'s ``BaggingClassifier``
    to the probabilistic regrsesion task.

    Parameters
    ----------
    estimator : skpro regressor, descendant of BaseProbaRegressor
        regressor to use in the bagging estimator
    n_estimators : int, default=10
        number of estimators in the sample for bagging
    n_samples : int or float, default=1.0
        The number of instances drawn from ``X`` in ``fit`` to train each clone
        If int, then indicates number of instances precisely
        If float, interpreted as a fraction, and rounded by ``ceil``
    n_features : int or float, default=1.0
        The number of features/variables drawn from ``X`` in ``fit`` to train each clone
        If int, then indicates number of instances precisely
        If float, interpreted as a fraction, and rounded by ``ceil``
    bootstrap : boolean, default=True
        whether samples/instances are drawn with replacement (True) or not (False)
    bootstrap_features : boolean, default=False
        whether features/variables are drawn with replacement (True) or not (False)
    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number generator;
        If ``RandomState`` instance, ``random_state`` is the random number generator;
        If None, the random number generator is the ``RandomState`` instance used
        by ``np.random``.

    Attributes
    ----------
    estimators_ : list of of skpro regressors
        clones of regressor in `estimator` fitted in the ensemble

    Examples
    --------
    >>> from skpro.regression.ensemble import BaggingRegressor
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg_mean = LinearRegression()
    >>> reg_proba = ResidualDouble(reg_mean)
    >>>
    >>> ens = BaggingRegressor(reg_proba, n_estimators=10)
    >>> ens.fit(X_train, y_train)
    BaggingRegressor(...)
    >>> y_pred = ens.predict_proba(X_test)
    """

    _tags = {"capability:missing": True}

    def __init__(
        self,
        estimator,
        n_estimators=10,
        n_samples=1.0,
        n_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        random_state=None,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state

        super().__init__()

        tags_to_clone = ["capability:missing"]
        self.clone_tags(estimator, tags_to_clone)

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
        estimator = self.estimator
        n_estimators = self.n_estimators
        n_samples = self.n_samples
        n_features = self.n_features
        bootstrap = self.bootstrap
        bootstrap_ft = self.bootstrap_features
        random_state = self.random_state
        np.random.seed(random_state)

        inst_ix = X.index
        col_ix = X.columns
        n = len(inst_ix)
        m = len(col_ix)

        if isinstance(n_samples, float):
            n_samples_ = ceil(n_samples * n)
        else:
            n_samples_ = n_samples

        if isinstance(n_features, float):
            n_features_ = ceil(n_features * m)
        else:
            n_features_ = n_features

        self.estimators_ = []
        self.cols_ = []

        for _i in range(n_estimators):
            esti = estimator.clone()
            row_iloc = pd.RangeIndex(n)
            row_ss = _random_ss_ix(row_iloc, size=n_samples_, replace=bootstrap)
            inst_ix_i = inst_ix[row_ss]
            col_ix_i = _random_ss_ix(col_ix, size=n_features_, replace=bootstrap_ft)

            # store column subset for use in predict
            self.cols_ += [col_ix_i]

            Xi = _subs_cols(X.loc[inst_ix_i], col_ix_i, reset_cols=bootstrap_ft)
            Xi = Xi.reset_index(drop=True)

            yi = y.loc[inst_ix_i].reset_index(drop=True)

            self.estimators_ += [esti.fit(Xi, yi)]

        return self

    def _predict_proba(self, X) -> np.ndarray:
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
        reset_cols = self.bootstrap_features
        Xis = [_subs_cols(X, col_ix_i, reset_cols) for col_ix_i in self.cols_]

        y_probas = [est.predict_proba(Xi) for est, Xi in zip(self.estimators_, Xis)]

        y_proba = Mixture(y_probas)

        return y_proba

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
        from sklearn.linear_model import LinearRegression

        from skpro.regression.residual import ResidualDouble

        regressor = ResidualDouble(LinearRegression())

        params1 = {"estimator": regressor}
        params2 = {
            "estimator": regressor,
            "n_samples": 0.5,
            "n_features": 0.5,
        }
        params3 = {
            "estimator": regressor,
            "n_samples": 7,
            "n_features": 2,
            "bootstrap": False,
            "bootstrap_features": True,
        }

        return [params1, params2, params3]


def _random_ss_ix(ix, size, replace=True):
    """Randomly uniformly sample indices from a list of indices."""
    a = range(len(ix))
    ixs = ix[np.random.choice(a, size=size, replace=replace)]
    return ixs


def _subs_cols(df, col_ix, reset_cols=False):
    """Subset columns of a DataFrame, with potential resetting of column index."""
    df_subset = df.loc[:, col_ix]
    if reset_cols:
        df_subset.columns = pd.RangeIndex(len(df_subset.columns))
    return df_subset
