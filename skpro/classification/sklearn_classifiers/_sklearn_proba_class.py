"""Adapter to sklearn multiclass classification using Histograms."""

__author__ = ["ShreeshaM07"]

import numpy as np
import pandas as pd

from skpro.distributions import Histogram
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class MulticlassSklearnProbaClassifier(BaseProbaRegressor):
    """A multiclass classifier fitting a histogram distribution."""

    _tags = {
        "authors": ["ShreeshaM07"],
        "maintainers": ["ShreeshaM07"],
        "capability:multioutput": False,
        "capability:missing": True,
    }

    def _convert_tuple_to_array(self, bins):
        bins_to_list = (bins[0], bins[1], bins[2])
        bins = []
        bin_width = (bins_to_list[1] - bins_to_list[0]) / bins_to_list[2]
        for b in range(bins_to_list[2]):
            bins.append(bins_to_list[0] + b * bin_width)
        bins.append(bins_to_list[1])
        return bins

    def __init__(self, clf, bins=(-500, 500, 10)):
        self.clf = clf
        self.bins = bins
        if isinstance(bins, tuple):
            bins = self._convert_tuple_to_array(bins)
        self._bins = bins

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
        from warnings import warn

        from numpy.lib.stride_tricks import sliding_window_view
        from sklearn import clone

        self.clf_ = clone(self.clf)
        bins = self._bins
        self._y_cols = y.columns

        # Generate class names based on bins
        classes_ = [f"class{i}" for i in range(len(bins) - 1)]

        class_series = pd.cut(y.iloc[:, 0], bins=bins, labels=classes_, right=True)
        y_binned = pd.DataFrame(class_series, columns=self._y_cols)

        X = prep_skl_df(X)
        y_binned = prep_skl_df(y_binned)

        if isinstance(y_binned, pd.DataFrame) and len(y_binned.columns) == 1:
            y_binned = y_binned.iloc[:, 0]
        elif len(y_binned.shape) > 1 and y_binned.shape[1] == 1:
            y_binned = y_binned[:, 0]

        self.clf_.fit(X, y_binned)
        self.classes_ = self.clf_.classes_
        classes_ = self.classes_

        if len(bins) != len(classes_) + 1:
            warn(
                f"len of `bins` is {len(bins)} != len of classes {len(classes_)}+1."
                " Ensure the bins has all the bin boundaries resulting in"
                " number of bins + 1 elements."
            )
            bins = np.arange(len(classes_) + 1)

        bins_hist = sliding_window_view(bins, window_shape=2)

        # maps the bin boundaries [bin start,bin end] to the classes
        class_bin_map_ = {}
        for i in range(len(bins_hist)):
            class_bin_map_[classes_[i]] = bins_hist[i]
        self.class_bin_map_ = class_bin_map_

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
        from warnings import warn

        X = prep_skl_df(X)
        bins = self._bins
        classes_ = self.classes_

        if len(bins) != len(classes_) + 1:
            warn(
                f"len of `bins` is {len(bins)} != len of classes {len(classes_)}+1."
                " Ensure the bins has all the bin boundaries resulting in"
                " number of bins + 1 elements."
            )
            bins = np.arange(len(classes_) + 1)

        y_pred_proba = self.clf_.predict_proba(X)

        # map classes probabilities/bin_mass to class names
        classes_proba_ = pd.DataFrame(y_pred_proba, columns=classes_)
        self.classes_proba_ = classes_proba_

        if len(X) == 1:
            bin_mass = y_pred_proba[0]
            pred_proba = Histogram(bins=bins, bin_mass=bin_mass)
            return pred_proba

        # converting it to a 2D shape
        bin_mass = np.array([y_pred_proba])
        # Reshape and swap axes to get the desired structure
        bin_mass = bin_mass.swapaxes(0, 1).reshape(-1, 1, bin_mass.shape[-1])

        bins = np.array([[bins]] * len(X))

        pred_proba = Histogram(
            bins=bins, bin_mass=bin_mass, index=X.index, columns=self._y_cols
        )
        return pred_proba

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
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier

        param1 = {"clf": DecisionTreeClassifier(), "bins": (-500, 500, 4)}
        param2 = {"clf": GaussianNB()}

        return [param1, param2]
