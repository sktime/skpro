# Example: using KDE and histogram with UnconditionalDistfitRegressor
import numpy as np
from skpro.regression.unconditional_distfit import UnconditionalDistfitRegressor

X = np.random.randn(80, 2)
y = np.random.randn(80)

# KDE baseline
reg_kde = UnconditionalDistfitRegressor(fit_kde=True)
reg_kde.fit(X, y)
dist_kde = reg_kde.predict_proba(X)
print('KDE baseline mean:', dist_kde.mean())

# Histogram baseline
reg_hist = UnconditionalDistfitRegressor(fit_histogram=True)
reg_hist.fit(X, y)
dist_hist = reg_hist.predict_proba(X)
print('Histogram baseline mean:', dist_hist.mean())
