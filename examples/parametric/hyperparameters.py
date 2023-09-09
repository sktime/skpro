# LEGACY MODULE - TODO: remove or refactor
from sklearn.datasets.base import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from skpro.parametric import ParametricEstimator
from skpro.parametric.estimators import Constant

model = ParametricEstimator(point=RandomForestRegressor(), std=Constant("mean(y)"))

# Initiate GridSearch meta-estimator
parameters = {"point__max_depth": [None, 5, 10, 15]}
clf = GridSearchCV(model, parameters)

# Optimize hyperparameters
X, y = load_diabetes(return_X_y=True)
clf.fit(X, y)

print("Best score is %f for parameter: %s" % (clf.best_score_, clf.best_params_))
# >>> Best score is -4.058729 for parameter: {'point__max_depth': 15}
