# -*- coding: utf-8 -*-
"""Base class for probabilistic regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import pandas as pd

from skpro.base import BaseEstimator
from skpro.utils.validation._dependencies import _check_estimator_deps


class BaseProbaRegressor(BaseEstimator):
    """Base class for probabilistic supervised regressors."""

    _tags = {
        "estimator_type": "regressor",
        "capability:multivariate": False,
        "capability:missing": True,
    }

    def __init__(self, index=None, columns=None):

        self.index = index
        self.columns = columns

        super(BaseProbaRegressor, self).__init__()
        _check_estimator_deps(self)

    def fit(self, X, y):
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

        Returns
        -------
        self : reference to self
        """
        X, y = self._check_X_y(X, y)

        # set fitted flag to True
        self._is_fitted = True

        return self._fit(X, y)

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
        raise NotImplementedError

    def predict(self, X):
        """Predict labels for data from features.

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
        y : pandas DataFrame, same length as `X`
            labels predicted for `X`
        """
        X = self._check_X(X)

        y_pred = self._predict(X)
        return y_pred

    def _predict(self, X):
        """Predict labels for data from features.

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
        y : pandas DataFrame, same length as `X`
            labels predicted for `X`
        """
        raise NotImplementedError

    def predict_proba(self, X):
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

        y_pred = self._predict_proba(X)
        return y_pred

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
        raise NotImplementedError

    def _check_X_y(self, X, y):

        X = self._check_X(X)
        y = self._check_y(y)

        # input check X vs y
        if not len(X) == len(y):
            raise ValueError(
                f"X and y in fit of {self} must have same number of rows, "
                f"but X had {len(X)} rows, and y had {len(y)} rows"
            )

        return X, y

    def _check_X(self, X):
        # if we have seen X before, check against columns
        if hasattr(self, "_X_columns") and not (X.columns == self._X_columns).all():
            raise ValueError(
                "X in predict methods must have same columns as X in fit, "
                f"columns in fit were {self._X_columns}, "
                f"but in predict found X.columns = {X.columns}"
            )
        # if not, remember columns
        else:
            self._X_columns = X.columns

        return X

    def _check_y(self, y):
        if isinstance(y, pd.Series):
            y = pd.DataFrame(y)
        return y
