import pymc3 as pm

from ...base import ProbabilisticEstimator
from ..bridge import ecdf


class PyMC(ProbabilisticEstimator):

    class Distribution(ProbabilisticEstimator.Distribution):

        def __init__(self, estimator, X):
            super().__init__(estimator, X)
            self.ecdf_ = None

        def point(self):
            self.estimator.samples_["y_pred"].mean(axis=0)

        def std(self):
            self.estimator.samples_["y_pred"].std(axis=0)

        def _ecdf(self):
            if self.ecdf_ is None:
                self.ecdf_ = [
                    ecdf(self.estimator.samples_["y_pred"][index, :])
                    for index in range(len(self.X))
                ]

            return self.ecdf_

        def cdf(self, x):
            return [self._ecdf()[index](x) for index in range(len(self.X))]

    def __init__(self, model=None):
        self.model = model
        self.samples_ = None
        self.trace_ = None

    def fit(self, X, y):
        self.model = self.model(y)
        with self.model:
            trace = pm.sample(1000)

        self.trace_ = trace

        return self

    def predict(self, X):
        with self.model:
            samples = pm.sample_ppc(self.trace_, samples=500, size=len(X))

        self.samples_ = samples
        return super().predict(X)
