# skpro developers, BSD-3-Clause License (see LICENSE file)
"""Histogram Conditional Density Estimation Regressor."""

__author__ = ["amaydixit11"]

import numpy as np
import pandas as pd
from skpro.distributions import Histogram
from skpro.regression.base import BaseProbaRegressor

class HistogramCDERegressor(BaseProbaRegressor):
    """Histogram Conditional Density Estimation (CDE) Regressor.

    Implements nonparametric conditional density estimation by binning the
    input space (X) and calculating the target (y) histogram within each bin.

    Parameters
    ----------
    n_neighbors : int, optional (default=10)
        Number of nearest neighbors to use for defining the local neighborhood in X.
    n_bins_y : int, optional (default=10)
        Number of bins for the target variable y.

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.regression.nonparametric import HistogramCDERegressor
    >>> X_train = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    >>> y_train = pd.DataFrame({"y": [1, 2, 2, 3, 3]})
    >>> reg = HistogramCDERegressor(n_neighbors=2, n_bins_y=3)
    >>> reg.fit(X_train, y_train)
    HistogramCDERegressor(...)
    >>> X_test = pd.DataFrame({"x": [2.5]})
    >>> y_pred = reg.predict_proba(X_test)
    """

    _tags = {
        "authors": ["amaydixit11"],
        "capability:multioutput": False,
        "capability:missing": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, n_neighbors=10, n_bins_y=10):
        self.n_neighbors = n_neighbors
        self.n_bins_y = n_bins_y
        super().__init__()

    def _fit(self, X, y):
        """Fit regressor to training data."""
        from sklearn.neighbors import NearestNeighbors
        
        self._X_train = X
        self._y_train = y
        
        n_samples_train = len(self._X_train)
        k = min(n_samples_train, max(1, getattr(self, "n_neighbors", 10)))
        self._knn = NearestNeighbors(n_neighbors=k)
        self._knn.fit(self._X_train.to_numpy())
        
        return self

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features."""
        self.check_is_fitted()
        
        # 1. Define bins for y
        y_min = self._y_train.min().min()
        y_max = self._y_train.max().max()
        
        # Add small additive buffer to prevent degenerate bins
        eps = max(0.01 * abs(y_max), np.finfo(float).eps * 10)
        if eps == 0:
            eps = 1e-8
            
        y_min -= eps
        y_max += eps
        
        if y_min == y_max:
            y_min -= 1e-8
            y_max += 1e-8
        
        y_bins = np.linspace(y_min, y_max, self.n_bins_y + 1)
        
        # 2. For each query point, find "nearby" points in X using pre-fitted KNN
        _distances, indices = self._knn.kneighbors(X.to_numpy())
        
        bin_masses = []
        for idx in indices:
            y_local = self._y_train.iloc[idx].to_numpy().flatten()
            counts, _ = np.histogram(y_local, bins=y_bins)
            mass = counts / counts.sum() if counts.sum() > 0 else np.ones(self.n_bins_y) / self.n_bins_y
            bin_masses.append(mass)
            
        # Histogram distribution expects nested format for array distributions:
        # bins and bin_mass should be list of lists of arrays [n_instances][n_columns]
        bin_mass_nested = [[np.array(m)] for m in bin_masses]
        bins_nested = [[np.array(y_bins)] for _ in range(len(bin_masses))]
        
        return Histogram(
            bins=bins_nested,
            bin_mass=bin_mass_nested,
            index=X.index,
            columns=self._y_train.columns
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{"n_neighbors": 5, "n_bins_y": 10}, {"n_neighbors": 10, "n_bins_y": 5}]
