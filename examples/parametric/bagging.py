# LEGACY MODULE - TODO: remove or refactor
from sklearn.tree import DecisionTreeRegressor

from skpro.ensemble import BaggingRegressor as SkproBaggingRegressor
from skpro.metrics import log_loss as loss
from skpro.parametric import ParametricEstimator
from skpro.workflow.manager import DataManager


def prediction(model, data):
    return model.fit(data.X_train, data.y_train).predict(data.X_test)


data = DataManager("boston")
clf = DecisionTreeRegressor()

baseline_prediction = prediction(ParametricEstimator(point=clf), data)

skpro_bagging_prediction = prediction(
    SkproBaggingRegressor(ParametricEstimator(point=clf), n_estimators=10, n_jobs=-1),
    data,
)

l1, l2 = loss(data.y_test, baseline_prediction), loss(
    data.y_test, skpro_bagging_prediction
)

print("Baseline: ", l1)
print("Bagged model:", l2)
