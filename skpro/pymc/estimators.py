import numpy as np
from sklearn.base import BaseEstimator
import pymc3 as pm


def _default_predictive_model(y):
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=y.mean(), sd=1)
        sd = pm.HalfNormal("sd", sd=1)
        y_pred = pm.Normal("y_pred", mu=mu, sd=sd, observed=y)

    return model


class BayesianLinearRegression(BaseEstimator):

    def __init__(self, model=_default_predictive_model):
        """
        Estimator wrapper interfacing PyMC3's Bayesian Linear Regression
        to be used with the parametric probabilistic estimator

        :param model: Optional callable with signature (X, y) that returns
                      a trace with linear model variables ``alpha`` and ``betas``
                      for the model f(x) = alpha + beta * X
        :param param_based_prediction: If true, the paramete
        """
        self.model = model

    def fit(self, X, y):

        return self

    def predict(self, X, return_std=False):
        with self.model(X):
            trace = pm.sample(1000)
            samples = pm.sample_ppc(trace, samples=500, size=len(X))

        y_pred = samples["y_pred"].mean(axis=0)

        if return_std:
            std = samples["y_pred"].std(axis=0)
            return np.stack((y_pred, std), axis=1)

        return y_pred


def _default_estimation_model(X, y):
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=y.mean(), sd=10)
        betas = pm.Normal("betas", mu=0, sd=10, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sd=10)

        mu = alpha + pm.math.dot(betas, X.T)
        y_pred = pm.Normal("y_pred", mu=mu, sd=sigma, observed=y)

    return model


class BayesianLinearRegression(BaseEstimator):

    def __init__(self, model=_default_estimation_model):
        """
        Estimator wrapper interfacing PyMC3's Bayesian Linear Regression
        to be used with the parametric probabilistic estimator

        :param model: Optional callable with signature (X, y) that returns
                      a trace with linear model variables ``alpha`` and ``betas``
                      for the model f(x) = alpha + beta * X
        """
        self.model = model
        self.chain_ = None

    def fit(self, X, y):
        model = self.model(X, y)

        self.chain_ = pm.sample(1000)[100:]

        return self

    def predict(self, X, return_std=False):
        alpha_pred = self.chain_['alpha'].mean()
        betas_pred = self.chain_['betas'].mean(axis=0)

        y_pred = alpha_pred + np.dot(betas_pred, X.T)

        if return_std:
            # use simplified propagation of error
            # https://doi.org/10.6028%2Fjres.070c.025
            alpha_std = self.chain['alpha'].std()
            betas_std = self.chain['betas'].std(axis=0)
            std = np.sqrt(alpha_std**2 + np.dot(betas_std**2, X.T**2))

            return np.stack((y_pred, std), axis=1)

        return y_pred
