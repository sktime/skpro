from skpro.base import ProbabilisticEstimator
from .interface import InterfacePyMC


class PyMC(ProbabilisticEstimator):

    class Distribution(ProbabilisticEstimator.Distribution):

        def point(self):
            return self.estimator.pymc_model.samples().mean(axis=1)

        def std(self):
            return self.estimator.pymc_model.samples().std(axis=1)

        def cdf(self, x):
            return self.estimator.adapter.cdf(x)

        def pdf(self, x):
            return self.estimator.adapter.pdf(x)

    def __init__(self, pymc_model=None, adapter=None):
        if not issubclass(pymc_model.__class__, InterfacePyMC):
            raise TypeError('model has to be a subclass of skpro.pymc.interface. '
                            '%s given.' % pymc_model.__class__)

        self.pymc_model = pymc_model
        self.adapter = adapter

    def fit(self, X, y):
        # fit the PyMC model
        self.pymc_model.on_fit(X, y)

        return self

    def predict(self, X):
        # predict with PyMC model
        self.pymc_model.on_predict(X)

        # initialise adapter with samples
        self.adapter(self.pymc_model.samples())

        # return predicted distribution object
        return super().predict(X)
