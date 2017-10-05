import numpy as np

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays

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


def test_kernel_density_adapter():
    # Bayesian test samples
    N = 5
    inlet = get_bayesian_sample(N)

    # Initialise adapter
    adapter = KernelDensityAdapter()
    adapter(inlet)
    x = np.random.laplace(0, 15, N)

    # PDF
    pdf = adapter.pdf(x)
    assert type(pdf) == np.ndarray
    assert len(pdf) == len(inlet)
    assert np.abs(pdf.mean()) < 20

    # CDF
    cdf = adapter.cdf(x)
    assert type(cdf) == np.ndarray
    assert len(cdf) == len(inlet)
    assert np.isclose(1, adapter.cdf(1e10))
    assert np.isclose(0, adapter.cdf(-10e10))


def test_empirical_density_adapter():
    # Bayesian test samples
    N = 5
    inlet = get_bayesian_sample(N)

    # Initialise adapter
    adapter = EmpiricalDensityAdapter()
    adapter(inlet)
    x = np.random.laplace(0, 15, N)

    # PDF
    pdf = adapter.pdf(x)
    assert type(pdf) == np.ndarray
    assert len(pdf) == len(inlet)
    assert np.abs(pdf.mean()) < 20

    # CDF
    cdf = adapter.cdf(x)
    assert type(cdf) == np.ndarray
    assert len(cdf) == len(inlet)
    assert np.isclose(1, adapter.cdf(1e10))
    assert np.isclose(0, adapter.cdf(-10e10))