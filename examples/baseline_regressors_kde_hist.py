"""Example: unconditional norm and laplace baselines with distfit."""
import logging

import numpy as np

from skpro.regression.unconditional_distfit import UnconditionalDistfitRegressor

X = np.random.randn(80, 2)
y = np.random.randn(80)

# Distfit norm baseline
reg_norm = UnconditionalDistfitRegressor(distr_type="norm")
reg_norm.fit(X, y)
dist_norm = reg_norm.predict_proba(X)
logging.info("Norm baseline mean: %s", dist_norm.mean())

# Distfit laplace baseline
reg_laplace = UnconditionalDistfitRegressor(distr_type="laplace")
reg_laplace.fit(X, y)
dist_laplace = reg_laplace.predict_proba(X)
logging.info("Laplace baseline mean: %s", dist_laplace.mean())
