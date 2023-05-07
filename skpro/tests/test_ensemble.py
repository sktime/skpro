from sklearn.ensemble import BaggingRegressor as ClassicBaggingRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from skpro.ensemble import BaggingRegressor as SkproBaggingRegressor
from skpro.metrics import log_loss as loss
from skpro.parametric import ParametricEstimator
from skpro.workflow.manager import DataManager


def prediction(model, data):
    return model.fit(data.X_train, data.y_train).predict(data.X_test)


def test_bagging_wrapper():
    data = DataManager('boston')

    # Run classic sklearn bagging mechanism to ensure we have
    # the correct parameters in place
    baseline_classic = prediction(DecisionTreeRegressor(), data)

    bagged_classic = prediction(
        ClassicBaggingRegressor(
            DecisionTreeRegressor()
        ),
        data
    )
    #
    # # Does the bagging reduce the loss?
    assert mse(data.y_test, baseline_classic) > mse(data.y_test, bagged_classic)

    # Run corresponding skpro bagging mechanism

    # clf = DecisionTreeRegressor()
    #
    # baseline_prediction = prediction(
    #     ParametricEstimator(point=clf),
    #     data
    # )
    #
    # skpro_bagging_prediction = prediction(
    #     SkproBaggingRegressor(
    #         ParametricEstimator(point=clf),
    #         n_estimators=10,
    #         n_jobs=-1
    #     ),
    #     data
    # )
    #
    # l1, l2 = loss(data.y_test, baseline_prediction),\
    #          loss(data.y_test, skpro_bagging_prediction)
    #
    # # Does the bagging reduce the loss?
    # assert l1 > l2