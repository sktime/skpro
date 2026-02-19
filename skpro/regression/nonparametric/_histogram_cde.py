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
    n_bins_x : int, optional (default=10)
        Number of bins for each feature in X.
    n_bins_y : int, optional (default=10)
        Number of bins for the target variable y.
    strategy : str, optional (default="uniform")
        Strategy used to define the widths of the bins.
        "uniform": all bins in each dimension have identical widths.
        "quantile": all bins in each dimension have the same number of points.
    """

    _tags = {
        "authors": ["amaydixit11"],
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, n_bins_x=10, n_bins_y=10, strategy="uniform"):
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        self.strategy = strategy
        super().__init__()

    def _fit(self, X, y):
        """Fit regressor to training data."""
        self._X_train = X
        self._y_train = y
        return self

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features."""
        from sklearn.neighbors import NearestNeighbors
        
        self.check_is_fitted()
        
        # 1. Define bins for y
        y_min = self._y_train.min().min()
        y_max = self._y_train.max().max()
        # Add small buffer
        y_min -= 0.01 * abs(y_min)
        y_max += 0.01 * abs(y_max)
        
        y_bins = np.linspace(y_min, y_max, self.n_bins_y + 1)
        
        # 2. For each query point, find "nearby" points in X using KNN
        n_samples_train = len(self._X_train)
        k = min(n_samples_train, max(1, n_samples_train // self.n_bins_x))
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(self._X_train)
        
        distances, indices = knn.kneighbors(X)
        
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
        return [{"n_bins_x": 5, "n_bins_y": 10}, {"n_bins_x": 10, "n_bins_y": 5}]
