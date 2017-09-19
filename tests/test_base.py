#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays

from skpro.base import ProbabilisticEstimator


class TestEstimator(ProbabilisticEstimator):

    class Distribution(ProbabilisticEstimator.Distribution):

        def point(self):
            self.estimator.predict(self.X)

        def std(self):
            return np.std(self.point())

        def pdf(self, x):
            return x


@given(arrays(np.float, 3, elements=floats(0, 1)))
def test_distribution_is_returned(X):
    estimator = TestEstimator()
    y_pred = estimator.predict(X)

    assert issubclass(y_pred.__class__, ProbabilisticEstimator.Distribution)
