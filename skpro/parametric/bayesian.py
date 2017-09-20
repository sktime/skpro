import numpy as np
from sklearn.base import BaseEstimator
import pymc3 as pm


def default_linear_model(X, y):
    model = pm.Model()
    with model:
        # Priors for unknown model parameters
        alpha = pm.Normal("alpha", mu=y.mean(), sd=10)
        betas = pm.Normal("betas", mu=0, sd=10, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sd=10)

        mu = alpha + pm.math.dot(betas, X.T)
        likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=y)

        step = pm.NUTS()
        trace = pm.sample(1000, step)

    return trace


class BayesianLinearRegression(BaseEstimator):

    def __init__(self, model=None):
        """
        Estimator wrapper interfacing PyMC3's Bayesian Linear Regression
        to be used with the parametric probabilistic estimator

        :param model: Optional callable with signature (X, y) that returns
                      a trace with linear model variables ``alpha`` and ``betas``
                      for the model f(x) = alpha + beta * X
        """
        if model is None:
            model = default_linear_model
        self.model = model
        self.chain = None

    def fit(self, X, y):
        trace = self.model(X, y)
        self.chain = trace[100:]

        return self

    def predict(self, X, return_std=False):
        alpha_pred = self.chain['alpha'].mean()
        betas_pred = self.chain['betas'].mean(axis=0)

        y_pred = alpha_pred + np.dot(betas_pred, X.T)

        if return_std:
            # use simplified propagation of error
            # https://doi.org/10.6028%2Fjres.070c.025
            alpha_std = self.chain['alpha'].std()
            betas_std = self.chain['betas'].std(axis=0)
            std = np.sqrt(alpha_std**2 + np.dot(betas_std**2, X.T**2))

            return np.stack((y_pred, std), axis=1)

        return y_pred
