#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from skpro.base import ProbabilisticEstimator, vectorvalued


class TestEstimator(ProbabilisticEstimator):

    class Distribution(ProbabilisticEstimator.Distribution):

        def point(self):
            return self.X[0]*10

        @vectorvalued
        def std(self):
            # returns a vector rather than a point
            return self.X[:, 0]/10

        def pdf(self, x):
            return -self.X[0]*x


def test_distribution_bracket_notation():
    estimator = TestEstimator()
    X = np.array([np.ones(3)*i for i in range(5)])
    y_pred = estimator.predict(X)

    # probabilistic estimator?
    assert issubclass(y_pred.__class__, ProbabilisticEstimator.Distribution)

    # does the replication works?
    assert issubclass(y_pred[1:3].__class__, ProbabilisticEstimator.Distribution)

    # does the __len__ reflect subsets?
    assert len(y_pred[0]) == 1
    assert len(y_pred[1:3]) == 2
    assert len(y_pred[:]) == len(y_pred._X)

    x = np.ones((5,)) * 4

    # MODE: elementwise

    # 0-dim, one dist, one point
    np.testing.assert_array_equal(y_pred[2].pdf(1), np.array([-2.]))
    # 0-dim, more dist than points
    np.testing.assert_array_equal(y_pred[1:4].pdf(7), np.array([-7., ]))

    # 1-dim, one dist, many points
    np.testing.assert_array_equal(y_pred[2].pdf(x), np.ones((5)) * -8.)
    # 1-dim, less dist than points
    np.testing.assert_array_equal(y_pred[2:4].pdf(x), np.array([ -8., -12.,  -8., -12.,  -8.]))
    # 1-dim, equal
    np.testing.assert_array_equal(y_pred[2:4].pdf(x[:2]), np.array([-8., -12.]))

    # MODE: batch

    # 0-dim, one dist, one point
    np.testing.assert_array_equal(y_pred[2, 'batch'].pdf(1), np.array([[-2.]]))
    # 0-dim, more dist than points
    np.testing.assert_array_equal(y_pred[1:4, 'batch'].pdf(7), np.array([[-7.], [-14.], [-21.]]))

    # 1-dim, one dist, many points
    np.testing.assert_array_equal(y_pred[2, 'batch'].pdf(x), [np.ones((5)) * -8.])
    # 1-dim, less dist than points
    np.testing.assert_array_equal(y_pred[2:4, 'batch'].pdf(x),
                                  [np.ones((5)) * -8., np.ones((5)) * -12.])
    # full batch notation
    np.testing.assert_array_equal(y_pred['batch'].pdf(1),
                                  -np.arange(5)[:, np.newaxis])


def test_interface_vectorization():
    estimator = TestEstimator()
    X = np.array([np.ones(3) * i for i in range(5)])
    y_pred = estimator.predict(X)

    # point interface
    np.testing.assert_array_equal(y_pred.point(), np.arange(5) * 10)
    # test vectorvalued decorator
    np.testing.assert_array_equal(y_pred.std(), np.arange(5) / 10)
    # lp2 integration
    np.testing.assert_array_equal(y_pred.lp2(), np.zeros(5))


def test_numeric_emulation():
    estimator = TestEstimator()
    A = np.array([np.ones(3) * i for i in range(5)])
    y_pred_1 = estimator.predict(A)
    B = np.array([-np.ones(3) * i for i in range(5)])
    y_pred_2 = estimator.predict(B)

    # only elementwise operation
    with pytest.raises(TypeError):
        float(y_pred_1)

    # type conversion
    assert float(y_pred_1[2]) == 20.0
    assert int(y_pred_1[3]) == 30

