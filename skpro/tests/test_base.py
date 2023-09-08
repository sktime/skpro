#!/usr/bin/env python
# LEGACY MODULE - TODO: remove or refactor

import numpy as np
import pytest

from skpro.base.old_base import ProbabilisticEstimator, vectorvalued


class EstimatorForTesting(ProbabilisticEstimator):
    def __init__(self):
        self.debug = dict()

    def debug_count(self, key):
        if key not in self.debug or not isinstance(self.debug[key], int):
            self.debug[key] = 1

        self.debug[key] += 1

    class Distribution(ProbabilisticEstimator.Distribution):
        def point(self):
            self.estimator.debug_count("point")
            return self.X[0] * 10

        @vectorvalued
        def std(self):
            self.estimator.debug_count("std")
            # returns a vector rather than a point
            return self.X[:, 0] / 10

        def pdf(self, x):
            self.estimator.debug_count("pdf")
            return -self.X[0] * x

        def lp2(self):
            x = 1
            return self[self.index].pdf(x) ** 2


def test_distribution_bracket_notation():
    estimator = EstimatorForTesting()
    X = np.array([np.ones(3) * i for i in range(5)])
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
    assert y_pred[2].pdf(1) == -2.0
    assert y_pred[3].pdf(2) == -6.0
    # 0-dim, more dist than points
    np.testing.assert_array_equal(y_pred[1:4].pdf(7), np.array([-7.0, -14.0, -21.0]))

    # 1-dim, one dist, many points
    np.testing.assert_array_equal(y_pred[2].pdf(x), np.ones((5)) * -8.0)
    # 1-dim, less dist than points
    np.testing.assert_array_equal(
        y_pred[2:4].pdf(x), np.array([-8.0, -12.0, -8.0, -12.0, -8.0])
    )
    # 1-dim, equal
    np.testing.assert_array_equal(y_pred[2:4].pdf(x[:2]), np.array([-8.0, -12.0]))

    # MODE: batch

    # 0-dim, one dist, one point
    assert y_pred[2, "batch"].pdf(1) == -2.0
    assert y_pred[3, "batch"].pdf(2) == -6.0
    # 0-dim, more dist than points
    np.testing.assert_array_equal(
        y_pred[1:4, "batch"].pdf(7), np.array([-7.0, -14.0, -21.0])
    )

    # 1-dim, one dist, many points
    np.testing.assert_array_equal(y_pred[2, "batch"].pdf(x), np.ones((5)) * -8.0)
    # 1-dim, less dist than points
    np.testing.assert_array_equal(
        y_pred[2:4, "batch"].pdf(x), [np.ones((5)) * -8.0, np.ones((5)) * -12.0]
    )
    # full batch notation
    np.testing.assert_array_equal(y_pred["batch"].pdf(1), -np.arange(5))


def test_interface_vectorization():
    estimator = EstimatorForTesting()
    X = np.array([np.ones(3) * i for i in range(5)])
    y_pred = estimator.predict(X)

    # point interface
    np.testing.assert_array_equal(y_pred.point(), np.arange(5) * 10)
    # test vectorvalued decorator
    np.testing.assert_array_equal(y_pred.std(), np.arange(5) / 10)
    # lp2 integration
    lp2 = y_pred.lp2()
    assert len(lp2) == 5
    assert lp2[0] == 0.0


def test_numeric_emulation():
    estimator = EstimatorForTesting()
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


def test_numpy_compatibility():
    estimator = EstimatorForTesting()

    A = np.array([np.ones(3) * i for i in range(5)])
    y_pred = estimator.predict(A)

    assert np.mean(np.std(y_pred)) == 0.2

    assert np.mean(np.mean(y_pred)) == 20
