"""Dummy time series regressor."""

__author__ = ["julian-fong"]
__all__ = ["DummyRegressor"]

import numpy as np

from skpro.regression.base import BaseProbaRegressor

# import pandas as pd


class DummyRegressor(BaseProbaRegressor):
    """DummyRegressor makes predictions that ignore the input features.

    This regressor serves as a simple baseline to compare against other more
    complex regressors.
    The specific behavior of the baseline is selected with the ``strategy``
    parameter.

    All strategies make predictions that ignore the input feature values passed
    as the ``X`` argument to ``fit`` and ``predict``. The predictions, however,
    typically depend on values observed in the ``y`` parameter passed to ``fit``.

    Function-identical to ``sklearn.dummy.DummyRegressor``, which is called
    inside.

    Parameters
    ----------
    strategy : one of ["empirical", "normal"] default="empirical"
        Strategy to use to generate predictions.

        * "empirical":
        * "normal": always predicts the median of the training set

    """

    def __init__(self, strategy="empirical"):
        self.strategy = strategy

        super().__init__()

    def _fit(self, X, y) -> np.ndarray:
        """Fit the dummy regressor.

        Parameters
        ----------
        X : sktime-format pandas dataframe with shape(n,d),
        or numpy ndarray with shape(n,d,m)
        y : array-like, shape = [n_instances] - the target values

        Returns
        -------
        self : reference to self.
        """
        self.sklearn_dummy_regressor.fit(np.zeros(X.shape), y)
        return self

    def _predict(self, X) -> np.ndarray:
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n, d)

        Returns
        -------
        y : predictions of target values for X, np.ndarray
        """
        return self.sklearn_dummy_regressor.predict(np.zeros(X.shape))
