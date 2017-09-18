import numpy as np
from sklearn.base import BaseEstimator
import pymc3 as pm


class BayesianLinearRegression(BaseEstimator):

    def __init__(self):
        self.pymc_model = pm.Model()
        self.chain = None

    def fit(self, X, y):

        with self.pymc_model:
            # Priors for unknown model parameters
            alpha = pm.Normal("alpha", mu=y.mean(), sd=10)
            betas = pm.Normal("betas", mu=0, sd=10, shape=X.shape[1])
            sigma = pm.HalfNormal("sigma", sd=10)

            mu = alpha + pm.math.dot(betas, X.T)
            likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=y)

            step = pm.NUTS()
            trace = pm.sample(1000, step)

        self.chain = trace[100:]

        return self

    def predict(self, X):
        alpha_pred = self.chain['alpha'].mean()
        betas_pred = self.chain['betas'].mean(axis=0)

        y_pred = alpha_pred + np.dot(betas_pred, X.T)

        return y_pred