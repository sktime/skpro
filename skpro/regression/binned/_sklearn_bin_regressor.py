"""Reduction to classification using sklearn classifiers fit on binned data."""

__author__ = ["ShreeshaM07"]

import numpy as np
import pandas as pd

from skpro.distributions import Histogram
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class HistBinnedProbaRegressor(BaseProbaRegressor):
    """A binned probabilistic regressor fitting a histogram distribution.

    It is a probabilistic regressor that fits a Histogram Distribution
    by presenting binned outcomes to a probabilistic sklearn classifier.
    It can be used for predicting the class that a set of X belongs to and
    predict_proba can be used to represent the predicted probabilites of
    each class for respective values in X in the form of a Histogram
    Distribution.

    The ``bins`` will be used to bin the ``y`` values in fit into the respective
    bins. It then uses these bins as the classes for the classifier and predicts
    the probabilites for each class.

    Note: Ensure the ``y`` values while calling ``fit`` are within the the ``bins``
    range. If it is not then it will be internally replaced to move to the
    closest bin.

    Parameters
    ----------
    clf : instance of a sklearn classifier
        Classifier to wrap, must have ``predict`` and ``predict_proba``.
    bins : int or 1D array of float, default: 10

        * If ``int`` then it will be considered as the number of bins.
        * Else if it is an array then it will be used as the bin boundaries.
          If the requirement is ``n`` bins then the ``len(bins)`` must be ``n+1``.

    Attributes
    ----------
    classes_ : np.array
        Contains the names of the classes that it was fit on.
    class_bin_map_ : dict
        The key contains the class name assigned to the bin.
        It maps the key (which indicates the ``i``th bin) to the respective
        bin's boundaries np.array([bins[i],bins[i+1]]).
    classes_proba_ : pd.DataFrame
        Contains the class probabilites.

    Examples
    --------
    >>> from skpro.regression.binned import HistBinnedProbaRegressor
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> hist_reg = HistBinnedProbaRegressor(RandomForestClassifier(), bins=5)
    >>> hist_reg.fit(X_train, y_train)
    HistBinnedProbaRegressor(...)
    >>>
    >>> y_pred = hist_reg.predict(X_test)
    >>> y_pred_proba = hist_reg.predict_proba(X_test)
    >>> y_pred_int = hist_reg.predict_interval(X_test)
    """

    _tags = {
        "authors": ["ShreeshaM07"],
        "maintainers": ["ShreeshaM07"],
        "capability:multioutput": False,
        "capability:missing": True,
    }

    def __init__(self, clf, bins=10):
        self.clf = clf
        self.bins = bins

        super().__init__()

    def _bins_int_arr(self, bins, y):
        y = np.array(y).flatten()
        start = min(y) * 0.999
        stop = max(y) * 1.001
        bins = np.linspace(start=start, stop=stop, num=bins + 1)
        return bins

    def _y_bins_compatiblity(self, y, bins, _y_cols):
        y = np.array(y).flatten()
        upper_y = bins[-1] - 1e-9
        lower_y = bins[0] + 1e-9
        y = np.where(y <= bins[0], lower_y, y)
        y = np.where(y >= bins[-1], upper_y, y)
        y = pd.DataFrame(y, columns=_y_cols)
        return y

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
        bins = self.bins
        self._y_cols = y.columns

        # create bins array in case of bins being an `int`
        if isinstance(bins, int) or isinstance(bins, np.integer):
            bins = self._bins_int_arr(bins, y)

        # check if y values are within bins range
        # if not move it to the closest bin.
        y = self._y_bins_compatiblity(y, bins, self._y_cols)

        # in case of int it will be internally replaced in fit
        self._bins = bins

        # Generate class names based on bins
        class_bins = [f"class{i}" for i in range(len(bins) - 1)]
        self._class_bins = class_bins

        if len(bins) != len(class_bins) + 1:
            warn(
                f"len of `bins` is {len(bins)} != len of classes {len(class_bins)}+1."
                " Ensure the bins has all the bin boundaries resulting in"
                " number of bins + 1 elements."
            )

        bins_hist = sliding_window_view(bins, window_shape=2)
        # maps the bin boundaries [bin start,bin end] to the classes
        class_bin_map_ = {}
        for i in range(len(bins_hist)):
            class_bin_map_[class_bins[i]] = bins_hist[i]
        self.class_bin_map_ = class_bin_map_

        # bins the y values into classes.
        class_series = pd.cut(y.iloc[:, 0], bins=bins, labels=class_bins, right=True)
        y_binned = pd.DataFrame(class_series, columns=self._y_cols)

        X = prep_skl_df(X)
        y_binned = prep_skl_df(y_binned)

        if isinstance(y_binned, pd.DataFrame) and len(y_binned.columns) == 1:
            y_binned = y_binned.iloc[:, 0]
        elif len(y_binned.shape) > 1 and y_binned.shape[1] == 1:
            y_binned = y_binned[:, 0]

        self.clf_.fit(X, y_binned)
        self.classes_ = self.clf_.classes_

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
        from warnings import warn

        X = prep_skl_df(X)
        bins = self._bins
        classes_ = self.classes_
        class_bins = self._class_bins

        if len(bins) != len(class_bins) + 1:
            warn(
                f"len of `bins` is {len(bins)} != len of classes {len(class_bins)}+1."
                " Ensure the bins has all the bin boundaries resulting in"
                " number of bins + 1 elements."
            )

        y_pred_proba = self.clf_.predict_proba(X)
        # map classes probabilities/bin_mass to class names
        classes_proba_ = pd.DataFrame(y_pred_proba, columns=classes_)

        # Identify missing classes
        missing_classes = set(class_bins) - set(classes_)
        if missing_classes:
            # Add missing classes with 0 values
            for missing_class in missing_classes:
                classes_proba_[missing_class] = 0
            # Sort columns based on the numerical part of the class names
            # in order to match with the bins while calling Histogram distribution
            classes_proba_ = classes_proba_.reindex(
                sorted(classes_proba_.columns, key=lambda x: int(x[5:])), axis=1
            )

        self.classes_proba_ = classes_proba_
        y_pred_proba = np.array(classes_proba_)

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
        from sklearn.semi_supervised import LabelSpreading
        from sklearn.tree import DecisionTreeClassifier

        param1 = {"clf": DecisionTreeClassifier(), "bins": 4}
        param2 = {"clf": GaussianNB()}
        params3 = {"clf": LabelSpreading(), "bins": [20, 80, 160, 250, 300, 380, 420]}

        return [param1, param2, params3]
