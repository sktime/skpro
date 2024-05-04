"""Probabilistic predictiins by point prediction."""

__author__ = ["fkiraly"]
__all__ = ["DeltaPointRegressor"]

import pandas as pd
from sklearn import clone

from skpro.distributions.delta import Delta
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class DeltaPointRegressor(BaseProbaRegressor):
    """Delta distribution prediction baseline regressor.

    This regressor turns an ``sklearn`` point prediction regressor into a probabilistic
    regressor by taking the point prediction as a predictive delta distribution.

    That is, on ``predict_proba``, a delta distribution with location
    identical to the regressor's ``predict`` is returned.

    Parameters
    ----------
    estimator : sklearn regressor
        regressor to use in the bootstrap

    Attributes
    ----------
    estimator_ : fitted sklearn regressor
        clones of regressor ``estimator``, fitted to the data

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg_tabular = LinearRegression()
    >>>
    >>> reg_proba = DeltaPointRegressor(reg_tabular)
    >>> reg_proba.fit(X_train, y_train)
    DeltaPointRegressor(...)
    >>> y_pred = reg_proba.predict_proba(X_test)
    """

    _tags = {"authors": "fkiraly", "capability:missing": True}

    def __init__(self, estimator):
        self.estimator = estimator

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
        self._cols = y.columns
        estimator = self.estimator

        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        # fit a clone of the estimator
        est_ = clone(estimator)
        self.estimator_ = est_.fit(X, y)
        return self

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
        cols = self._cols

        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        # predict point predictions
        y_pred = self.estimator_.predict(X)

        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=cols, index=X.index)

        # create delta distribution
        y_proba = Delta(c=y_pred)
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
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        params1 = {"estimator": LinearRegression()}
        params2 = {"estimator": RandomForestRegressor()}

        return [params1, params2]
