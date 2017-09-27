from sklearn.datasets import load_boston
import pymc3 as pm

from skpro.bayesian.pymc import PyMC

from skpro.parametric import ParamtericEstimator
from skpro.bayesian.pymc.estimators import BayesianLinearRegression, BayesianLinearEstimation
from skpro.metrics import rank_probability_loss, linearized_log_loss
from skpro.workflow.manager import DataManager
from skpro.workflow.table import Table


def pymc_model(y):
    model = pm.Model()
    with model:
        mu = pm.Normal("mu", mu=y.mean(), sd=1)
        sd = pm.HalfNormal("sd", sd=1)
        y_pred = pm.Normal("y_pred", mu=mu, sd=sd, observed=y)

    return model


X, y = load_boston(return_X_y=True)
data = DataManager(X, y, name='Boston')

tbl = Table().info()\
    .cv(data, linearized_log_loss)\
    .cv(data, rank_probability_loss)

tbl.print([
    PyMC(model=pymc_model),
    ParamtericEstimator(),
    ParamtericEstimator(point_std=BayesianLinearRegression()),
    ParamtericEstimator(point_std=BayesianLinearEstimation())
])
