"""Deterministic regression reduction baseline.

Outputs Gaussian (or Laplace) with mean=prediction, var=training sample var.
"""

import numpy as np

from skpro.distributions.laplace import Laplace
from skpro.distributions.normal import Normal
from skpro.regression.base import BaseProbaRegressor


class DeterministicReductionRegressor(BaseProbaRegressor):
    """
    Wraps a deterministic regressor to output a Gaussian or Laplace.

    The output has mean=prediction, var=training sample var.

    References
    ----------
    - Gaussian Processes for State Space Models and Change Point Detection
      (Turner, 2011 thesis). https://mlg.eng.cam.ac.uk/pub/pdf/Tur11.pdf
    - A Probabilistic View of Linear Regression
      (Bishop, PRML; Keng, 2016; various tutorials).
    - mlr3proba and related probabilistic ML frameworks.
    - Efficient and Distance-Aware Deep Regressor for Uncertainty Quantification
      (Bui et al., 2024).
      https://proceedings.mlr.press/v238/manh-bui24a/manh-bui24a.pdf

    Examples
    --------
    >>> from skpro.regression.deterministic_reduction import \
    ...     DeterministicReductionRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> import numpy as np
    >>> X = np.random.randn(100, 2)
    >>> y = np.random.randn(100)
    >>> reg = DeterministicReductionRegressor(LinearRegression(), distr_type='gaussian')
    >>> reg.fit(X, y)
    >>> dist = reg.predict_proba(X)
    >>> print(dist.mean().shape)
    (100,)
    """

    def __init__(self, regressor, distr_type="gaussian"):
        self.regressor = regressor
        self.distr_type = distr_type
        super().__init__()

    def _fit(self, X, y, C=None):
        # Ensure X and y are DataFrames with string column names
        import pandas as pd
        from sklearn.base import clone
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        X.columns = [str(col) for col in X.columns]
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        y = y.copy()
        y.columns = [str(col) for col in y.columns]
        self._X_cols = X.columns
        self._y_cols = y.columns
        self._X_index = X.index
        self._y_index = y.index
        # Clone the regressor to avoid mutating the parameter
        self.regressor_ = clone(self.regressor)
        self.regressor_ = self.regressor_.fit(
            X, y.values.ravel() if y.shape[1] == 1 else y
        )
        y_arr = y.values.flatten()
        self.train_mean_ = np.mean(y_arr)
        self.train_var_ = np.var(y_arr)
        return self

    def _predict_proba(self, X):
        import pandas as pd
        # Ensure X is a DataFrame with string column names
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._X_cols)
        X = X.copy()
        X.columns = [str(col) for col in X.columns]
        mean_pred = self.regressor_.predict(X)
        # Ensure output shape matches y
        if mean_pred.ndim == 1:
            mean_pred = mean_pred.reshape(-1, 1)
        # Return distribution with correct index/columns
        if self.distr_type == "gaussian":
            return Normal(
                mu=mean_pred,
                sigma=np.sqrt(self.train_var_),
                index=X.index,
                columns=self._y_cols,
            )
        if self.distr_type == "laplace":
            # Laplace scale = sqrt(var/2)
            return Laplace(
                mu=mean_pred,
                scale=np.sqrt(self.train_var_ / 2),
                index=X.index,
                columns=self._y_cols,
            )
        raise ValueError(f"Unknown distr_type: {self.distr_type}")

    def get_params(self, deep=True):
        # Only return true hyperparameters, not fitted attributes
        return {"regressor": self.regressor, "distr_type": self.distr_type}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for automated tests."""
        from sklearn.linear_model import LinearRegression

        return [
            {"regressor": LinearRegression(), "distr_type": "gaussian"},
            {"regressor": LinearRegression(), "distr_type": "laplace"},
        ]
