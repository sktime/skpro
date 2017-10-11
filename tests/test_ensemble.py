import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor as ClassicBaggingRegressor
from sklearn.metrics import mean_squared_error as mse

from skpro.workflow.manager import DataManager
from skpro.parametric import ParametricEstimator
from skpro.ensemble import BaggingRegressor
from skpro.metrics import linearized_log_loss as loss

def test_bagging_wrapper():
    data = DataManager('boston')

    def prediction(model):
        return model.fit(data.X_train, data.y_train).predict(data.X_test)

    # classical sklearn bagging mechanism to ensure we have the correct
    # parameters in place

    baseline_classic = prediction(DecisionTreeRegressor())
    bagged_classic = prediction(ClassicBaggingRegressor(
        DecisionTreeRegressor(),
        max_samples=0.5,
        max_features=0.5,
        bootstrap=False,
        n_estimators=100
    ))

    # does the bagging reduce the loss?
    assert mse(data.y_test, baseline_classic) > mse(data.y_test, bagged_classic)

    # corresponding skpro bagging mechanism

    baseline = prediction(ParametricEstimator(point=DecisionTreeRegressor()))
    bagged = prediction(BaggingRegressor(
        ParametricEstimator(point=DecisionTreeRegressor()),
        max_samples=0.5,
        max_features=0.5,
        bootstrap=False,
        n_estimators=100
    ))

    # does the bagging reduce the loss?
    assert loss(data.y_test, baseline) > loss(data.y_test, bagged)