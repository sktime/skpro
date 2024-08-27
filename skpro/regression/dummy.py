"""Dummy time series regressor."""

__author__ = ["julian-fong"]
__all__ = ["DummyProbaRegressor"]

import numpy as np
import pandas as pd

from skpro.distributions.empirical import Empirical
from skpro.distributions.normal import Normal
from skpro.regression.base import BaseProbaRegressor


class DummyProbaRegressor(BaseProbaRegressor):
    """DummyProbaRegressor makes predictions that ignore the input features.

    This regressor serves as a simple baseline to compare against other more
    complex regressors.
    The specific behavior of the baseline is selected with the ``strategy``
    parameter.

    All strategies make predictions that ignore the input feature values passed
    as the ``X`` argument to ``fit`` and ``predict``. The predictions, however,
    typically depend on values observed in the ``y`` parameter passed to ``fit``.

    Parameters
    ----------
    strategy : one of ["empirical", "normal"] default="empirical"
        Strategy to use to generate predictions.

        * "empirical": always predicts the empirical unweighted distribution
            of the training labels
        * "normal": always predicts a normal distribution, with mean and variance
            equal to the mean and variance of the training labels

    Attributes
    ----------
    distribution_ : skpro.distribution
        Normal distribution or Empirical distribution, depending on chosen strategy.
        Scalar version of the distribution that is returned by ``predict_proba``.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["julian-fong"],
        "maintainers": ["julian-fong"],
        # estimator tags
        # --------------
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, strategy="empirical"):
        self.strategy = strategy
        super().__init__()

    def _fit(self, X, y):
        """Fit the dummy regressor.

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
        self._y = y
        self._y_columns = y.columns
        self._mu = np.mean(y.values)
        self._sigma = np.std(y.values)
        if self.strategy == "empirical":
            self.distribution_ = Empirical(y)
        if self.strategy == "normal":
            self.distribution_ = Normal(self._mu, self._sigma)

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n, d)

        Returns
        -------
        y : pandas DataFrame
            predictions of target values for X
        """
        X_ind = X.index
        X_n_rows = X.shape[0]
        y_pred = pd.DataFrame(
            np.ones(X_n_rows) * self._mu, index=X_ind, columns=self._y_columns
        )
        return y_pred

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
        X_ind = X.index
        X_n_rows = X.shape[0]
        y_pred = pd.DataFrame(
            np.ones(X_n_rows) * self._sigma, index=X_ind, columns=self._y_columns
        )
        return y_pred

    def _predict_proba(self, X):
        """Broadcast skpro distribution from fit onto labels from X.

        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n, d)

        Returns
        -------
        y : skpro.distribution, same length as `X`
            labels predicted for `X`
        """
        X_ind = X.index
        X_n_rows = X.shape[0]
        if self.strategy == "normal":
            # broadcast the mu and sigma from fit to the length of X
            mu = np.reshape((np.ones(X_n_rows) * self._mu), (-1, 1))
            sigma = np.reshape((np.ones(X_n_rows) * self._sigma), (-1, 1))
            pred_dist = Normal(mu=mu, sigma=sigma, index=X_ind, columns=self._y_columns)
            return pred_dist

        if self.strategy == "empirical":
            empr_df = pd.concat([self._y] * X_n_rows, keys=X_ind).swaplevel()
            pred_dist = Empirical(empr_df, index=X_ind, columns=self._y_columns)

            return pred_dist

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
        params1 = {}
        params2 = {"strategy": "normal"}

        return [params1, params2]
