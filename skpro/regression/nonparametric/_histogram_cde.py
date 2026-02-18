# skpro developers, BSD-3-Clause License (see LICENSE file)
"""Histogram Conditional Density Estimation Regressor."""

__author__ = ["amaydixit11"]

import numpy as np
import pandas as pd
from skpro.regression.base import BaseProbaRegressor

class HistogramCDERegressor(BaseProbaRegressor):
    """Histogram Conditional Density Estimation (CDE) Regressor."""

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
        self.check_is_fitted()
        pass

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{"n_bins_x": 5, "n_bins_y": 10}, {"n_bins_x": 10, "n_bins_y": 5}]
