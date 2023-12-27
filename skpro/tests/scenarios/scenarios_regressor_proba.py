"""Test scenarios for classification and regression.

Contains TestScenario concrete children to run in tests for classifiers/regressirs.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_regressor_proba"]

from inspect import isclass

import numpy as np
import pandas as pd

from skpro.base import BaseObject
from skpro.tests.scenarios.scenarios import TestScenario


class _ProbaRegressorTestScenario(TestScenario, BaseObject):
    """Generic test scenario for classifiers."""

    def get_args(self, key, obj=None, deepcopy_args=True):
        """Return args for key. Can be overridden for dynamic arg generation.

        If overridden, must not have any side effects on self.args
            e.g., avoid assignments args[key] = x without deepcopying self.args first

        Parameters
        ----------
        key : str, argument key to construct/retrieve args for
        obj : obj, optional, default=None. Object to construct args for.
        deepcopy_args : bool, optional, default=True. Whether to deepcopy return.

        Returns
        -------
        args : argument dict to be used for a method, keyed by `key`
            names for keys need not equal names of methods these are used in
                but scripted method will look at key with same name as default
        """
        PREDICT_LIKE_FUNCTIONS = ["predict", "predict_var", "predict_proba"]
        # use same args for predict-like functions as for predict
        if key in PREDICT_LIKE_FUNCTIONS:
            key = "predict"

        return super().get_args(key=key, obj=obj, deepcopy_args=deepcopy_args)

    def is_applicable(self, obj):
        """Check whether scenario is applicable to obj.

        Parameters
        ----------
        obj : class or object to check against scenario

        Returns
        -------
        applicable: bool
            True if self is applicable to obj, False if not
        """

        def get_tag(obj, tag_name):
            if isclass(obj):
                return obj.get_class_tag(tag_name)
            else:
                return obj.get_tag(tag_name)

        # applicable only if object is a BaseProbaRegressor
        if not get_tag(obj, "object_type") == "regressor_proba":
            return False

        # if X has missing data, applicable only if can handle missing data
        has_missing_data = self.get_tag("X_missing", False, False)
        if has_missing_data and not get_tag(obj, "capability:missing"):
            return False

        return True


# as functions to ensure we can move to fixture-like structure later
def _load_diabetes_small():
    """Load diabetes dataset, return first 50 samples."""
    import pandas as pd
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.iloc[:50]
    y = y.iloc[:50]
    y = pd.DataFrame(y)

    return X, y


def _get_Xy_traintest():
    """Get X, y train/test data for regression tests.

    X, y are pandas.DataFrame, y is univariate.
    X has homogeneous dtype, and no missing data.

    Returns
    -------
    X_train : pandas.DataFrame, training data
    X_test : pandas.DataFrame, test data
    y_train : pandas.DataFrame, training labels
    y_test : pandas.DataFrame, test labels
    """
    from sklearn.model_selection import train_test_split

    X, y = _load_diabetes_small()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = _get_Xy_traintest()
C_train = pd.DataFrame(
    np.random.choice([0, 1], size=len(y_train)),
    index=y_train.index,
    columns=y_train.columns,
)


class ProbaRegressorBasic(_ProbaRegressorTestScenario):
    """Fit/predict with multivariate pandas mtype (same dtype), and labels y."""

    _tags = {
        "X_univariate": False,
        "X_missing": False,
        "y_univariate": True,
        "is_enabled": True,
    }

    args = {
        "fit": {"X": X_train, "y": y_train},
        "predict": {"X": X_test},
    }
    default_method_sequence = ["fit", "predict", "predict_proba"]
    default_arg_sequence = ["fit", "predict", "predict"]


class ProbaRegressorSurvival(_ProbaRegressorTestScenario):
    """Fit/predict with survival data, including a censoring dataset."""

    _tags = {
        "X_univariate": False,
        "X_missing": False,
        "y_univariate": True,
        "is_enabled": True,
    }

    args = {
        "fit": {"X": X_train, "y": y_train, "C": C_train},
        "predict": {"X": X_test},
    }
    default_method_sequence = ["fit", "predict", "predict_proba"]
    default_arg_sequence = ["fit", "predict", "predict"]


# as a function to ensure we can move to fixture-like structure later
def _get_Xy_traintest_X_mixix_ynp():
    """Get X, y train/test data for regression tests.

    X is pandas.DataFrame, y is univariate numpy array.
    X has homogeneous dtype, and no missing data.

    X has mixed column index type (int and str).

    Returns
    -------
    X_train : pandas.DataFrame, training data
    X_test : pandas.DataFrame, test data
    y_train : pandas.DataFrame, training labels
    y_test : pandas.DataFrame, test labels
    """
    from sklearn.model_selection import train_test_split

    X, y = _load_diabetes_small()

    X.columns = ["age", "sex", "bmi", "bp", 1, 2, 3, 4, 5, 6]
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test


X_train_mc, X_test_mc, y_train_mc, y_test_mc = _get_Xy_traintest_X_mixix_ynp()


class ProbaRegressorXcolMixIxYnp(_ProbaRegressorTestScenario):
    """Fit/predict with multivariate pandas mtype, mixed col idx type."""

    _tags = {
        "X_univariate": False,
        "X_missing": False,
        "y_univariate": True,
        "is_enabled": True,
    }

    args = {
        "fit": {"X": X_train_mc, "y": y_train_mc},
        "predict": {"X": X_test_mc},
    }
    default_method_sequence = ["fit", "predict", "predict_proba"]
    default_arg_sequence = ["fit", "predict", "predict"]


scenarios_regressor_proba = [
    ProbaRegressorBasic,
    ProbaRegressorXcolMixIxYnp,
    ProbaRegressorSurvival,
]
