"""Implements target transformation pipeline element for probabilistic regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["arnavk23"]
__all__ = ["TargetTransform"]

from skpro.regression.base import BaseProbaRegressor
from skpro.regression.compose._ttr import TransformedTargetRegressor


class TargetTransform(BaseProbaRegressor):
    """TargetTransform pipeline for target variable transformation.

    Wraps a regressor and a transformer, applying the transformer to y
    during fit and inverse-transforming predictions.
    Uses TransformedTargetRegressor internally.

    Parameters
    ----------
    regressor : BaseProbaRegressor
        The probabilistic regressor to wrap.
    transformer : sklearn-like transformer
        The transformer to apply to the target variable.

    Examples
    --------
    >>> from skpro.regression.compose import TargetTransform
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sklearn.preprocessing import StandardScaler, MinMaxScaler
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> import pandas as pd
    >>> # Load data
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> y = pd.DataFrame(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> # Create a probabilistic regressor
    >>> reg = ResidualDouble.create_test_instance()
    >>> # Use StandardScaler for target transformation
    >>> ttr = TargetTransform(regressor=reg, transformer=StandardScaler())
    >>> ttr.fit(X_train, y_train)
    TargetTransform(...)
    >>> y_pred = ttr.predict(X_test)
    >>> y_pred_proba = ttr.predict_proba(X_test)
    >>> # Use MinMaxScaler for target transformation
    >>> ttr2 = TargetTransform(regressor=reg, transformer=MinMaxScaler())
    >>> ttr2.fit(X_train, y_train)
    TargetTransform(...)
    >>> y_pred2 = ttr2.predict(X_test)
    """

    _tags = {
        "capability:multioutput": True,
        "capability:missing": True,
    }

    def __init__(self, regressor, transformer):
        self.regressor = regressor
        self.transformer = transformer
        self._ttr = TransformedTargetRegressor(
            regressor=regressor, transformer=transformer
        )
        super().__init__()

    def _fit(self, X, y, C=None):
        self._ttr.fit(X, y, C=C)
        return self

    def _predict(self, X):
        return self._ttr.predict(X)

    def _predict_quantiles(self, X, alpha):
        return self._ttr.predict_quantiles(X, alpha)

    def _predict_interval(self, X, coverage):
        return self._ttr.predict_interval(X, coverage)

    def _predict_var(self, X):
        return self._ttr.predict_var(X)

    def _predict_proba(self, X):
        return self._ttr.predict_proba(X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for automated tests.

        Returns two parameter sets: one with StandardScaler, one with MinMaxScaler.
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        from skpro.regression.residual import ResidualDouble

        reg = ResidualDouble.create_test_instance()
        return [
            {"regressor": reg, "transformer": StandardScaler()},
            {"regressor": reg, "transformer": MinMaxScaler()},
        ]
