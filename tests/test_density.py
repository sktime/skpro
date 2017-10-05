import numpy as np

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays

from skpro.density import ecdf


@given(arrays(np.float, 10, elements=floats(0, 100)))
def test_ecdf_from_sample(sample):
    xs, ys = ecdf(sample)

    # correct mapping?
    assert len(xs) == len(ys)

    # is it monotone?
    assert np.array_equal(ys, sorted(ys))