# -*- coding: utf-8 -*-
import numpy as np

from sklearn.linear_model.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.stats import _weighted_percentile



class DummyRegressor(BaseEstimator, RegressorMixin):
 
    def __init__(self, strategy="mean", constant=None, quantile=None):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile
        

    def fit(self, X, y, sample_weight=None):
       
        allowed_strategies = ("mean", "median", "quantile", "constant", "variance")
        if self.strategy not in allowed_strategies:
            raise ValueError("Unknown strategy type: %s, expected one of %s."
                             % (self.strategy, allowed_strategies))

        y = check_array(y, ensure_2d=False)
        if len(y) == 0:
            raise ValueError("y must not be empty.")

        self.output_2d_ = y.ndim == 2
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]

        check_consistent_length(X, y, sample_weight)

        if self.strategy == "mean":
            self.constant_ = np.average(y, axis=0, weights=sample_weight)
            
        elif self.strategy == "variance":
            self.constant_ = np.var(y, axis=0)

        elif self.strategy == "median":
            if sample_weight is None:
                self.constant_ = np.median(y, axis=0)
            else:
                self.constant_ = [_weighted_percentile(y[:, k], sample_weight,
                                                       percentile=50.)
                                  for k in range(self.n_outputs_)]

        elif self.strategy == "quantile":
            if self.quantile is None or not np.isscalar(self.quantile):
                raise ValueError("Quantile must be a scalar in the range "
                                 "[0.0, 1.0], but got %s." % self.quantile)

            percentile = self.quantile * 100.0
            if sample_weight is None:
                self.constant_ = np.percentile(y, axis=0, q=percentile)
            else:
                self.constant_ = [_weighted_percentile(y[:, k], sample_weight,
                                                       percentile=percentile)
                                  for k in range(self.n_outputs_)]

        elif self.strategy == "constant":
            if self.constant is None:
                raise TypeError("Constant target value has to be specified "
                                "when the constant strategy is used.")

            self.constant = check_array(self.constant,
                                        accept_sparse=['csr', 'csc', 'coo'],
                                        ensure_2d=False, ensure_min_samples=0)

            if self.output_2d_ and self.constant.shape[0] != y.shape[1]:
                raise ValueError(
                    "Constant target value should have "
                    "shape (%d, 1)." % y.shape[1])

            self.constant_ = self.constant

        self.constant_ = np.reshape(self.constant_, (1, -1))
        return self


    def predict(self, X, return_std=False):
  
        check_is_fitted(self, "constant_")
        n_samples = _num_samples(X)

        y = np.full((n_samples, self.n_outputs_), self.constant_,
                    dtype=np.array(self.constant_).dtype)
        y_std = np.zeros((n_samples, self.n_outputs_))

        if self.n_outputs_ == 1 and not self.output_2d_:
            y = np.ravel(y)
            y_std = np.ravel(y_std)

        return (y, y_std) if return_std else y

    
    def score(self, X, y, sample_weight=None):
       
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        return super(DummyRegressor, self).score(X, y, sample_weight)
    
