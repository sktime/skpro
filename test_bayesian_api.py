import numpy as np
import pandas as pd
from skpro.regression.bayesian import BayesianConjugateLinearRegressor

# Fixed data and test point
X_initial = pd.DataFrame({"f": [1.0, 2.0]})
y_initial = pd.Series([2.0, 4.0])
X_new = pd.DataFrame({"f": [10.0, 20.0]})
y_new = pd.Series([100.0, 200.0])
X_test = pd.DataFrame({"f": [5.0]})

# Init with large variance to see the change
model = BayesianConjugateLinearRegressor(coefs_prior_cov=[[10.0]], noise_precision=1.0)

model.fit(X_initial, y_initial)
sig1 = model.predict_proba(X_test).sigma.values[0][0]

# Perform Update
model.update(X_new, y_new)
sig2 = model.predict_proba(X_test).sigma.values[0][0]

print(f"\nSigma 1: {sig1:.4f}")
print(f"Sigma 2: {sig2:.4f}")

if sig2 < sig1:
    print("SUCCESS: Uncertainty reduced.")