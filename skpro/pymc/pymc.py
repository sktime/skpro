from skpro.base import ProbabilisticEstimator
from .interface import PyMCInterface


class PyMC(ProbabilisticEstimator):

    class Distribution(ProbabilisticEstimator.Distribution):

        def point(self):
            return self.estimator.pymc_model.samples().mean(axis=1)

        def std(self):
            return self.estimator.pymc_model.samples().std(axis=1)

        def cdf(self, x):
            pass

        def pdf(self, x):
            pass

    def __init__(self, pymc_model=None, adapter=None):
        if not issubclass(pymc_model.__class__, PyMCInterface):
            raise TypeError('model has to be a subclass of skpro.pymc.interface. '
                            '%s given.' % pymc_model.__class__)

        self.pymc_model = pymc_model
        self.adapter = adapter

    def fit(self, X, y):
        self.pymc_model.on_fit(X, y)

        return self

    def predict(self, X):
        self.pymc_model.on_predict(X)

        return super().predict(X)
