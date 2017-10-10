import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from scipy.stats import norm

from skpro.density import ecdf, KernelDensityAdapter, EmpiricalDensityAdapter

np.random.seed(1)


def get_bayesian_sample(points, sample_size=500):
    """ Helper to generate bayesian samples as returned by Bayesian prediction methods

    Parameters
    ----------
    points          Number of data points (rows)
    sample_size     Number of sample elements for each data point (columns)

    Returns
    -------
    np.array    N x M matrix containing example bayesian samples for data point predictions
    """
    return np.array([
        np.random.normal(np.random.randint(0, 15), np.random.randint(0, 15), sample_size)
        for _ in range(points)
    ])


# --- TESTS ---------------------


@given(arrays(np.float, 10, elements=floats(0, 100)))
def test_ecdf_from_sample(sample):
    xs, ys = ecdf(sample)

    # correct mapping?
    assert len(xs) == len(ys)

    # is it monotone?
    assert np.array_equal(ys, sorted(ys))


@given(floats(-10, 10))
def test_kernel_density_adapter(x):
    # Bayesian test sample
    loc, scale = 5, 10
    sample = np.random.normal(loc=loc, scale=scale, size=500)

    # Initialise adapter
    adapter = KernelDensityAdapter()
    adapter(sample)

    # PDF
    pdf = adapter.pdf(x)
    assert isinstance(pdf, np.float)
    assert abs(pdf - norm.pdf(x, loc=loc, scale=scale)) < 0.01

    # CDF
    cdf = adapter.cdf(x)
    assert isinstance(cdf, np.float)
    # assert abs(cdf - norm.cdf(x, loc=5, scale=10)) < 0.1


@given(floats(-10, 10))
def test_empirical_density_adapter(x):
    # Bayesian test sample\
    loc, scale = 5, 10

    sample = np.random.normal(loc=loc, scale=scale, size=5000)

    # Initialise adapter
    adapter = EmpiricalDensityAdapter()
    adapter(sample)

    # PDF
    pdf = adapter.pdf(x)
    assert isinstance(pdf, float)
    # assert abs(pdf - norm.pdf(x, loc=loc, scale=scale)) < 0.1

    # CDF
    cdf = adapter.cdf(x)
    assert isinstance(cdf, float)
    assert abs(cdf - norm.cdf(x, loc=loc, scale=scale)) < 0.1
