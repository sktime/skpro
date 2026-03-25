# Example usage for baseline probabilistic regressors
import numpy as np
from sklearn.linear_model import LinearRegression
from skpro.regression.unconditional_distfit import UnconditionalDistfitRegressor
from skpro.regression.deterministic_reduction import DeterministicReductionRegressor

# Generate synthetic data
X = np.random.randn(100, 3)
y = 2 * X[:, 0] + np.random.randn(100)

# 1. Unconditional density baseline (featureless)
reg1 = UnconditionalDistfitRegressor()
reg1.fit(X, y)
dist1 = reg1.predict_proba(X)
print('UnconditionalDistfitRegressor mean:', dist1.mean())
print('Sample from unconditional:', dist1.sample(5))

# 2. Deterministic-style baseline (mean from regressor, constant variance)
reg2 = DeterministicReductionRegressor(LinearRegression(), distr_type='gaussian')
reg2.fit(X, y)
dist2 = reg2.predict_proba(X)
print('DeterministicReductionRegressor mean:', dist2.mean)
print('DeterministicReductionRegressor sigma:', dist2.sigma)
print('Sample from deterministic baseline:', dist2.sample(5))
