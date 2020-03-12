from sklearn.ensemble.bagging import BaggingRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from skpro.workflow.manager import DataManager
from skpro.distributions.distribution_base import DistributionBase
from skpro.estimators.parametric import ParametricEstimator
from skpro.ensemble import ProbabilisticBaggingRegressor


def prediction(estimator, data):
    return estimator.fit(data.X_train, data.y_train).predict(data.X_test)


def test_classic_bagging_wrapper():
    data = DataManager('boston')

    # Run classic sklearn bagging mechanism to ensure we have the correct parameters in place
    baseline_classic = prediction(DecisionTreeRegressor(), data)
    
    bagginEstimator = BaggingRegressor(DecisionTreeRegressor())
    bagged_classic = prediction(bagginEstimator, data)

    # Does the bagging reduce the loss?
    assert mse(data.y_test, baseline_classic) > mse(data.y_test, bagged_classic)


def test_probabilistic_bagging_wrapper():
    data = DataManager('boston')
    model = ParametricEstimator(LinearRegression(), LinearRegression())
    bagginEstimator = ProbabilisticBaggingRegressor(base_estimator = model)
    bagginEstimator.fit(data.X_train, data.y_train)
    y_hat = bagginEstimator.predict_proba(data.X_test)
    
    assert(isinstance(y_hat, DistributionBase))
    

    
if __name__ == "__main__":
     test_classic_bagging_wrapper()
     test_probabilistic_bagging_wrapper()
