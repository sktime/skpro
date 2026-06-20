"""Adapter to sklearn probabilistic classifiers."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly", "skpro-developers"]

import pandas as pd

from skpro.classification.base import BaseProbaClassifier
from skpro.distributions.discrete import Discrete
from skpro.utils.sklearn import prep_skl_df


class SklearnClassifierAdapter(BaseProbaClassifier):
    """Adapter to sklearn probabilistic classifiers.

    Wraps an sklearn classifier that can be queried for predict_proba,
    and constructs an skpro classifier from it.

    Parameters
    ----------
    estimator : sklearn classifier
        Estimator to wrap, must have ``predict_proba`` method.
    inner_type : str, one of "pd.DataFrame", "np.ndarray", default="pd.DataFrame"
        Type of X passed to ``fit`` and ``predict`` methods of the wrapped estimator.
        Type of y passed to ``fit`` method of the wrapped estimator.
    """

    _tags = {
        "capability:multioutput": False,
        "capability:missing": True,
    }

    def __init__(self, estimator, inner_type="pd.DataFrame"):
        self.estimator = estimator
        self.inner_type = inner_type
        super().__init__()

    def _coerce_inner(self, obj):
        """Coerce obj to type of inner_type."""
        obj = prep_skl_df(obj)
        if self.inner_type == "np.ndarray":
            obj = obj.to_numpy()
        return obj

    def _fit(self, X, y):
        """Fit classifier to training data."""
        from sklearn import clone

        self.estimator_ = clone(self.estimator)
        self._y_cols = y.columns

        X_inner = self._coerce_inner(X)
        y_inner = self._coerce_inner(y)

        if isinstance(y_inner, pd.DataFrame) and len(y_inner.columns) == 1:
            y_inner = y_inner.iloc[:, 0]
        elif len(y_inner.shape) > 1 and y_inner.shape[1] == 1:
            y_inner = y_inner[:, 0]

        self.estimator_.fit(X_inner, y_inner)
        return self

    def _predict(self, X):
        """Predict labels for data from features."""
        X_inner = self._coerce_inner(X)
        y_pred = self.estimator_.predict(X_inner)
        y_pred_df = pd.DataFrame(y_pred, index=X.index, columns=self._y_cols)
        return y_pred_df

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features."""
        X_inner = self._coerce_inner(X)
        
        y_prob = self.estimator_.predict_proba(X_inner)
        classes = self.estimator_.classes_
        
        # Wrapped as Discrete distribution
        return Discrete(
            probabilities=y_prob, 
            classes=classes, 
            index=X.index, 
            columns=self._y_cols
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        param1 = {"estimator": LogisticRegression(max_iter=10)}
        param2 = {"estimator": RandomForestClassifier(n_estimators=5)}

        return [param1, param2]
