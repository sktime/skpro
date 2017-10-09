import pymc3 as pm
from sklearn.datasets import load_boston

from skpro.metrics import rank_probability_loss, linearized_log_loss
from skpro.parametric import ParamtericEstimator
from skpro.workflow.manager import DataManager

from skpro.pymc import PyMC


def pymc_model(y):
    model = pm.Model()
    with model:
        mu = pm.Normal("mu", mu=y.mean(), sd=1)
        sd = pm.HalfNormal("sd", sd=1)
        y_pred = pm.Normal("y_pred", mu=mu, sd=sd, observed=y)

    return model


def test_construct_estimator(self):
    X, y = load_boston(return_X_y=True)
    data = DataManager(X, y, name='Boston')

    model = PyMC()
    y_pred = model.fit(data.X_train, data.y_train).predict(data.X_test)




