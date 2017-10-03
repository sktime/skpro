import numpy as np

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays

from skpro.utils import ecdf


@given(arrays(np.float, 50, elements=floats(0, 100)))
def test_ecdf_from_sample(sample):
    xs, ys = ecdf(sample, return_func=False)

    assert len(xs) == len(ys)
    # todo: assert xs, ys is the correct empirical cdf of the sample

    ecdf_func = ecdf(sample, return_func=True)
    # todo: assert ecdf is the correct ecdf step function