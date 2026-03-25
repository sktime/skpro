"""
Deterministic regression reduction baseline: outputs Gaussian (or Laplace) with mean=prediction, var=training sample var.
"""

import numpy as np
from skpro.regression.base import BaseProbaRegressor
from skpro.distributions.normal import Normal
from skpro.distributions.laplace import Laplace


class DeterministicReductionRegressor(BaseProbaRegressor):
    """
    Wraps a deterministic regressor to output a Gaussian or Laplace with mean=prediction, var=training sample var.

    References
    ----------
    - Gaussian Processes for State Space Models and Change Point Detection (Turner, 2011 thesis).
      https://mlg.eng.cam.ac.uk/pub/pdf/Tur11.pdf
    - A Probabilistic View of Linear Regression (Bishop, PRML; Keng, 2016; various tutorials).
    - mlr3proba and related probabilistic ML frameworks.
    - Efficient and Distance-Aware Deep Regressor for Uncertainty Quantification (Bui et al., 2024).
      https://proceedings.mlr.press/v238/manh-bui24a/manh-bui24a.pdf
    """

    def __init__(self, regressor, distr_type='gaussian'):
        self.regressor = regressor
        self.distr_type = distr_type
        super().__init__()

    def _fit(self, X, y, C=None):
        self.regressor_ = self.regressor.fit(X, y)
        y_arr = y.values.flatten() if hasattr(y, 'values') else np.asarray(y).flatten()
        self.train_mean_ = np.mean(y_arr)
        self.train_var_ = np.var(y_arr)
        return self

    def _predict_proba(self, X):
        mean_pred = self.regressor_.predict(X)
        if self.distr_type == 'gaussian':
            return Normal(mu=mean_pred, sigma=np.sqrt(self.train_var_))
        elif self.distr_type == 'laplace':
            # Laplace scale = sqrt(var/2)
            return Laplace(mu=mean_pred, scale=np.sqrt(self.train_var_/2))
        else:
            raise ValueError(f"Unknown distr_type: {self.distr_type}")
