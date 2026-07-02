"""Example usage for baseline probabilistic regressors."""
import logging

import numpy as np
from sklearn.linear_model import LinearRegression

from skpro.regression.deterministic_reduction import DeterministicReductionRegressor
from skpro.regression.unconditional_distfit import UnconditionalDistfitRegressor

# Generate synthetic data
X = np.random.randn(100, 3)
y = 2 * X[:, 0] + np.random.randn(100)

# 1. Unconditional density baseline (featureless)
reg1 = UnconditionalDistfitRegressor()
reg1.fit(X, y)
dist1 = reg1.predict_proba(X)
logging.info("UnconditionalDistfitRegressor mean: %s", dist1.mean())
logging.info("Sample from unconditional: %s", dist1.sample(5))

# 2. Deterministic-style baseline (mean from regressor, constant variance)
reg2 = DeterministicReductionRegressor(LinearRegression(), distr_type="gaussian")
reg2.fit(X, y)
dist2 = reg2.predict_proba(X)
logging.info("DeterministicReductionRegressor mean: %s", dist2.mean)
logging.info("DeterministicReductionRegressor sigma: %s", dist2.sigma)
logging.info("Sample from deterministic baseline: %s", dist2.sample(5))
