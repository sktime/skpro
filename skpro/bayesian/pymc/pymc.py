import pymc3 as pm
import numpy as np
from sklearn.neighbors import KernelDensity

from ...base import ProbabilisticEstimator
from ..bridge import ecdf


class PyMC(ProbabilisticEstimator):

    class Distribution(ProbabilisticEstimator.Distribution):

        def __init__(self, estimator, X):
            super().__init__(estimator, X)
            self.ecdf_ = None
            self.kde_ = None

        def point(self):
            self.estimator.samples_["y_pred"].mean(axis=0)

        def std(self):
            self.estimator.samples_["y_pred"].std(axis=0)

        def sample(self, index):
            return self.estimator.samples_["y_pred"][index, :]

        def _kde(self):
            if self.kde_ is None:

                self.kde_ = [
                    KernelDensity().fit(self.sample(index)[:, np.newaxis])
                    for index in range(len(self.X))
                ]

            return self.kde_

        def pdf(self, x):
            return [
                np.exp(self._kde()[index].score_samples(x[:, np.newaxis]))[0]
                for index in range(len(self.X))
            ]

            #kde.fit(x[:, np.newaxis])
            # score_samples() returns the log-likelihood of the samples
            #return np.exp(log_pdf)

        def _ecdf(self):
            if self.ecdf_ is None:
                self.ecdf_ = [
                    ecdf(self.sample(index))
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
