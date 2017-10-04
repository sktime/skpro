import pymc3 as pm
from sklearn.datasets import load_boston

from skpro.metrics import rank_probability_loss, linearized_log_loss
from skpro.parametric import ParamtericEstimator
from skpro.pymc import PyMC, PyMCPlugAndPlay
from skpro.workflow.manager import DataManager
from skpro.workflow.table import Table



X, y = load_boston(return_X_y=True)
data = DataManager(X, y, name='Boston')


def model_definition(model, X, y, X_shape):
    with model:
        # Priors
        alpha = pm.Normal('alpha', mu=y.mean(), sd=10)
        betas = pm.Normal('beta', mu=0, sd=10, shape=X_shape[1])
        sigma = pm.HalfNormal('sigma', sd=1)

        # Model (has to define y_pred)
        mu = alpha + pm.math.dot(betas, X.T)
        y_pred = pm.Normal("y_pred", mu=mu, sd=sigma, observed=y)


model = PyMC(pymc_model=PyMCPlugAndPlay(model_definition))

y_pred = model.fit(data.X_train, data.y_train).predict(data.X_test)


from matplotlib import pyplot

pyplot.scatter(y_pred, data.y_test)

pyplot.show()