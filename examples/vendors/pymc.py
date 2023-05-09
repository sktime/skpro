# -*- coding: utf-8 -*-
import pymc3 as pm

from skpro.base import BayesianVendorEstimator
from skpro.metrics import log_loss
from skpro.vendors.pymc import PymcInterface
from skpro.workflow.manager import DataManager

# Define the model using PyMC's syntax


def pymc_linear_regression(model, X, y):
    """Defines a linear regression model in PyMC

    Parameters
    ----------
    model: PyMC model
    X: Features
    y: Labels

    The model must define a ``y_pred`` model variable that represents the prediction target
    """

    with model:
        # Priors
        alpha = pm.Normal("alpha", mu=y.mean(), sd=10)
        betas = pm.Normal("beta", mu=0, sd=10, shape=X.get_value(borrow=True).shape[1])
        sigma = pm.HalfNormal("sigma", sd=1)

        # Model (defines y_pred)
        mu = alpha + pm.math.dot(betas, X.T)
        y_pred = pm.Normal("y_pred", mu=mu, sd=sigma, observed=y)


# Plug the model definition into the PyMC interface

model = BayesianVendorEstimator(
    model=PymcInterface(model_definition=pymc_linear_regression)
)


# Run prediction, print and plot the results

data = DataManager("boston")
y_pred = model.fit(data.X_train, data.y_train).predict(data.X_test)
print("Log loss: ", log_loss(data.y_test, y_pred, return_std=True))

# Plot the performance
import sys

sys.path.append("../")
import skpro.examples.utils

utils.plot_performance(data.y_test, y_pred)
