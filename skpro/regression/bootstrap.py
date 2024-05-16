"""Probabilistic regression by bootstrap."""

__author__ = ["fkiraly"]
__all__ = ["BootstrapRegressor"]

import numpy as np
import pandas as pd
from sklearn import clone

from skpro.distributions.empirical import Empirical
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.numpy import flatten_to_1D_if_colvector
from skpro.utils.sklearn import prep_skl_df


class BootstrapRegressor(BaseProbaRegressor):
    """Bootstrap ensemble of a tabular regressor.

    Fits ``n_estimators`` clones of an skpro regressor on
    datasets which are bootstrap sub-samples, i.e.,
    independent row samples with replacement.

    On ``predict_proba``, an empirical distribution with the bootstrap
    sample is returned.

    The estimator allows to choose sample sizes for instances, variables,
    and whether sampling is with or without replacement.

    Direct generalization of ``sklearn``'s ``BaggingClassifier``
    to the probabilistic regression task.

    Parameters
    ----------
    estimator : sklearn regressor
        regressor to use in the bootstrap
    n_bootstrap_samples : int, default=100
        The number of bootstrap samples drawn
        If int, then indicates number of instances precisely
        Note: this is not the same as the size of each bootstrap sample.
        The size of the bootstrap sample is always equal to X.
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
    >>> from skpro.regression.bootstrap import BootstrapRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg_tabular = LinearRegression()
    >>>
    >>> reg_proba = BootstrapRegressor(reg_tabular)
    >>> reg_proba.fit(X_train, y_train)
    BootstrapRegressor(...)
    >>> y_pred = reg_proba.predict_proba(X_test)
    """

    _tags = {"authors": "fkiraly", "capability:missing": True}

    def __init__(
        self,
        estimator,
        n_bootstrap_samples=100,
        random_state=None,
    ):
        self.estimator = estimator
        self.n_bootstrap_samples = n_bootstrap_samples
        self.random_state = random_state

        super().__init__()

        # todo: find the equivalent tag in sklearn for missing data handling
        # tags_to_clone = ["capability:missing"]
        # self.clone_tags(estimator, tags_to_clone)

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
        n_bootstrap_samples = self.n_bootstrap_samples
        np.random.seed(self.random_state)

        inst_ix = X.index
        n = len(inst_ix)

        self.estimators_ = []
        self._cols = y.columns

        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        for _i in range(n_bootstrap_samples):
            esti = clone(estimator)
            row_iloc = pd.RangeIndex(n)
            row_ss = _random_ss_ix(row_iloc, size=n, replace=True)
            inst_ix_i = inst_ix[row_ss]

            Xi = X.loc[inst_ix_i]
            Xi = Xi.reset_index(drop=True)

            yi = y.loc[inst_ix_i].values
            yi = flatten_to_1D_if_colvector(yi)

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
        cols = self._cols

        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        y_preds = [est.predict(X) for est in self.estimators_]

        def _coerce_df(x):
            if not isinstance(x, pd.DataFrame):
                x = pd.DataFrame(x, columns=cols, index=X.index)
            return x

        y_preds = [_coerce_df(x) for x in y_preds]

        y_pred_df = pd.concat(y_preds, axis=0, keys=range(len(y_preds)))

        y_proba = Empirical(y_pred_df)
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

        params1 = {"estimator": LinearRegression()}
        params2 = {
            "estimator": LinearRegression(),
            "n_bootstrap_samples": 10,
        }

        return [params1, params2]


def _random_ss_ix(ix, size, replace=True):
    """Randomly uniformly sample indices from a list of indices."""
    a = range(len(ix))
    ixs = ix[np.random.choice(a, size=size, replace=replace)]
    return ixs
