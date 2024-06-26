"""Adapter to sklearn multiclass classification using Histograms."""

__author__ = ["ShreeshaM07"]

import numpy as np
import pandas as pd

from skpro.distributions import Histogram
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class SklearnProbaClassifier(BaseProbaRegressor):
    """A multiclass classifier fitting a histogram distribution."""

    _tags = {
        "authors": ["ShreeshaM07"],
        "maintainers": ["ShreeshaM07"],
        "capability:multioutput": False,
        "capability:missing": True,
    }

    def __init__(self, clf, bins=10):
        self.clf = clf
        if isinstance(bins, int) or isinstance(bins, np.integer):
            bins = np.arange(bins + 1)
        self.bins = bins

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
        from sklearn import clone

        self.clf_ = clone(self.clf)
        self._y_cols = y.columns

        X = prep_skl_df(X)
        y = prep_skl_df(y)

        if isinstance(y, pd.DataFrame) and len(y.columns) == 1:
            y = y.iloc[:, 0]
        elif len(y.shape) > 1 and y.shape[1] == 1:
            y = y[:, 0]

        self.clf_.fit(X, y)
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
        X = prep_skl_df(X)
        y_pred = self.clf_.predict(X)
        y_pred_df = pd.DataFrame(y_pred, index=X.index, columns=self._y_cols)
        return y_pred_df

    def _predict_var(self, X):
        """Compute/return variance predictions.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        pred_var : pd.DataFrame
            Column names are exactly those of ``y`` passed in ``fit``.
            Row index is equal to row index of ``X``.
            Entries are variance prediction, for var in col index.
            A variance prediction for given variable and fh index is a predicted
            variance for that variable and index, given observed data.
        """
        X = prep_skl_df(X)
        _, y_std = self.clf_.predict(X, return_std=True)
        y_std = pd.DataFrame(y_std, index=X.index, columns=self._y_cols)
        y_var = y_std**2
        return y_var

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
        X = prep_skl_df(X)
        # print(X)
        y_pred_proba = self.clf_.predict_proba(X)
        bin_mass = y_pred_proba[0]
        # print(bin_mass)
        self.classes_ = self.clf_.classes_
        classes_ = self.classes_
        bins = self.bins
        # print(bins)
        if len(bins) != len(classes_) + 1:
            bins = np.arange(len(classes_) + 1)
        pred_proba = Histogram(bins=bins, bin_mass=bin_mass)
        return pred_proba
