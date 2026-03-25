# Benchmark script for baseline probabilistic regressors
import numpy as np
from sklearn.linear_model import LinearRegression
from skpro.regression.unconditional_distfit import UnconditionalDistfitRegressor
from skpro.regression.deterministic_reduction import DeterministicReductionRegressor
from skpro.metrics import PinballLoss

# Generate synthetic data
X = np.random.randn(200, 5)
y = 3 * X[:, 0] - 2 * X[:, 1] + np.random.randn(200)

# Split
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Baseline 1: Unconditional
reg1 = UnconditionalDistfitRegressor()
reg1.fit(X_train, y_train)
dist1 = reg1.predict_proba(X_test)

# Baseline 2: Deterministic reduction
reg2 = DeterministicReductionRegressor(LinearRegression(), distr_type='gaussian')
reg2.fit(X_train, y_train)
dist2 = reg2.predict_proba(X_test)

# Evaluate pinball loss at alpha=0.1, 0.5, 0.9
alphas = [0.1, 0.5, 0.9]
for alpha in alphas:
    loss1 = PinballLoss(alpha=alpha)(y_test, dist1)
    loss2 = PinballLoss(alpha=alpha)(y_test, dist2)
    print(f"Alpha={alpha}: UnconditionalDistfitRegressor pinball loss={loss1:.4f}, DeterministicReductionRegressor pinball loss={loss2:.4f}")
