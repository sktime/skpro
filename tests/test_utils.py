import numpy as np

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays

from skpro.utils import ecdf


@given(arrays(np.float, 10, elements=floats(0, 100)))
def test_ecdf_from_sample(sample):
    xs, ys = ecdf(sample, return_func=False)
    ecdf_func = ecdf(sample, return_func=True)

    # correct mapping?
    assert len(xs) == len(ys)

    # is it monotone?
    assert np.array_equal(ys, sorted(ys))

    # function wrapper working?
    assert ecdf_func(xs[0]) == ys[0]