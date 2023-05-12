# -*- coding: utf-8 -*-
import numpy as np

from skpro.base.old_base import ProbabilisticEstimator, vectorvalued
from skpro.regression.density import DensityAdapter, KernelDensityAdapter
from skpro.utils.utils import ensure_existence


class DensityBaseline(ProbabilisticEstimator):
    class Distribution(ProbabilisticEstimator.Distribution):
        @vectorvalued
        def point(self):
            return np.ones((len(self.X),)) * self.estimator.training_mean_

        @vectorvalued
        def std(self):
            return np.ones((len(self.X),)) * self.estimator.training_std_

        def cdf(self, x):
            ensure_existence(self.estimator.adapter.cdf)

            return self.estimator.adapter.cdf(x)

        def pdf(self, x):
            ensure_existence(self.estimator.adapter.pdf)

            return self.estimator.adapter.pdf(x)

    def __init__(self, adapter=None):
        if adapter is None:
            adapter = KernelDensityAdapter()

        if not issubclass(adapter.__class__, DensityAdapter):
            raise ValueError(
                "adapter has to be a subclass of skpro.density.DensityAdapter"
                "%s given." % adapter.__class__
            )

        self.adapter = adapter
        self.training_mean_ = None
        self.training_std_ = None

    def fit(self, X, y):
        # Use the labels to estimate the density
        self.adapter(y)
        self.training_mean_ = np.mean(y)
        self.training_std_ = np.std(y)

        return self
