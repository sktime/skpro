from sklearn.base import BaseEstimator
from pymc3 import Model, Normal, HalfNormal, find_MAP, NUTS, sample, summary, Slice
import numpy as np
from skpro.base import describe


class BayesianLinearRegression(BaseEstimator):

    def __init__(self, mode='MAP'):
        """
       
        :param mode: MAP, 
         maximum a posteriori (MAP)
         Broyden–Fletcher–Goldfarb–Shanno (BFGS) optimization algorithm to find the maximum of the log-posterior
        """
        self.mode = mode
        self.pymc_model = Model()
        self.dimensions = None
        self.map_estimate = None

    def fit(self, X, y):
        if self.map_estimate is not None:
            return self

        self.dimensions = X.shape[1]

        with self.pymc_model:
            # Priors for unknown model parameters
            alpha = Normal('alpha', mu=0, sd=10)
            beta = Normal('beta', mu=0, sd=10, shape=self.dimensions)
            sigma = HalfNormal('sigma', sd=1)

            # Expected value of outcome
            mu = alpha
            for i in range(self.dimensions):
                mu += beta[i] * X[:, i]

            # Likelihood (sampling distribution) of observations
            Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=y)

        self.map_estimate = find_MAP(model=self.pymc_model)

        return self

    def predict(self, X):
        y_pred = self.map_estimate['alpha'] * np.ones((len(X),))
        for i in range(self.dimensions):
            y_pred += self.map_estimate['beta'][i] * X[:, i]

        return y_pred

    def description(self):
        return 'BLR(MAP)'


class BayesianLinearRegressionSampling(BaseEstimator):

    def __init__(self):
        self.pymc_model = Model()
        self.dimensions = None
        self.trace = None

    def fit(self, X, y):
        if self.trace is not None:
            return self

        self.dimensions = X.shape[1]

        with self.pymc_model:
            # Priors for unknown model parameters
            alpha = Normal('alpha', mu=0, sd=10)
            beta = Normal('beta', mu=0, sd=10, shape=self.dimensions)
            sigma = HalfNormal('sigma', sd=1)

            # Expected value of outcome
            mu = alpha
            for i in range(self.dimensions):
                mu += beta[i] * X[:, i]

            # Likelihood (sampling distribution) of observations
            Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=y)

            # obtain starting values via MAP
            start = find_MAP()

            # draw 5000 posterior samples
            self.trace = sample(500, start=start)

        return self

    def predict(self, X):
        y_pred = self.trace.get_values('alpha').mean(0) * np.ones((len(X),))
        for i in range(self.dimensions):
            y_pred += self.trace.get_values('beta').mean(0)[i] * X[:, i]

        return y_pred

    def predict_std(self, X):
        # TODO
        return np.ones((len(X), ))

    def description(self):
        return 'BLR(SAMPLE)'