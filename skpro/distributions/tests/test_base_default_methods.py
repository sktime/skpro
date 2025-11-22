"""Test class for default methods.

This is not for direct use, but for testing whether the defaulting in various
methods works.

Testing works via TestAllDistributions which discovers the classes in
here, executes the public methods in interface conformance tests,
which in turn triggers the fallback defaults.
"""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
import pytest
from scipy.special import erfinv

from skpro.distributions.base import BaseDistribution
from skpro.tests.test_switch import run_test_module_changed
from skpro.utils.estimator_checks import check_estimator


# normal distribution with exact implementations removed
class _DistrDefaultMethodTester(BaseDistribution):
    """Tester distribution for default methods."""

    _tags = {
        "capabilities:approx": ["pdfnorm", "mean", "var", "energy", "log_pdf", "cdf"],
        "capabilities:exact": ["pdf", "ppf"],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma

        super().__init__(index=index, columns=columns)

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Parameters
        ----------
        p : 2D np.ndarray, same shape as ``self``
            values to evaluate the ppf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            ppf values at the given points
        """
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        icdf_arr = mu + sigma * np.sqrt(2) * erfinv(2 * p - 1)
        return icdf_arr

    def _pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pdf values at the given points
        """
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        pdf_arr = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        pdf_arr = pdf_arr / (sigma * np.sqrt(2 * np.pi))
        return pdf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "sigma": 1}
        params2 = {
            "mu": 0,
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": 1, "sigma": 2}
        return [params1, params2, params3]


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_base_default():
    """Test default methods.

    The _DistributionDefaultMethodTester class is not detected
    by TestAllDistributions (it is private), so we need to test it explicitly.

    check_estimator invokes a TestAllDistributions call.
    """
    check_estimator(_DistrDefaultMethodTester, raise_exceptions=True)


# normal distribution with only sample method
class _DistrDefaultMethodTesterOnlySample(BaseDistribution):
    """Tester distribution for default methods."""

    _tags = {
        "capabilities:approx": [
            "pdfnorm",
            "mean",
            "var",
            "energy",
            "log_pdf",
            "cdf",
            "pdf",
            "ppf",
        ],
        "capabilities:exact": [],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma

        super().__init__(index=index, columns=columns)

    def sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None
            number of samples to draw from the distribution

        Returns
        -------
        pd.DataFrame
            samples from the distribution
        """
        if self.ndim == 0:
            if n_samples is None:
                return np.random.normal(loc=self.mu, scale=self.sigma)
            res = np.random.normal(loc=self.mu, scale=self.sigma, size=n_samples)
            return pd.DataFrame(res)
        # else: self.ndim is 2
        if n_samples is None:
            res_shape = self.shape
            vals = np.random.normal(loc=self.mu, scale=self.sigma, size=res_shape)
            return pd.DataFrame(vals, index=self.index, columns=self.columns)
        # else: n_samples is given
        res_shape = (n_samples * self.shape[0], self.shape[1])
        vals = np.random.normal(loc=self.mu, scale=self.sigma, size=res_shape)
        multiindes = pd.MultiIndex.from_product(np.arange(n_samples), self.index)
        return pd.DataFrame(vals, index=multiindes, columns=self.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "sigma": 1}
        params2 = {
            "mu": 0,
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": 1, "sigma": 2}
        return [params1, params2, params3]


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_base_default_minimal_cdf():
    """Test default cdf method."""
    minimal_n = _DistrDefaultMethodTesterOnlySample(mu=0, sigma=1)
    assert minimal_n.cdf(0) < minimal_n.cdf(100)


class _CompositeDistributionTester(BaseDistribution):
    """Composite distribution for testing _subset_param with distribution parameters."""

    _tags = {
        "capabilities:approx": [],
        "capabilities:exact": ["mean"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "composite",
        "broadcast_init": "on",
    }

    def __init__(self, distribution, scalar_param=1.0, index=None, columns=None):
        """Initialize composite distribution.

        Parameters
        ----------
        distribution : BaseDistribution
            Inner distribution parameter.
        scalar_param : float, default=1.0
            A scalar parameter for testing.
        index : pd.Index, optional
            Index for the distribution.
        columns : pd.Index, optional
            Columns for the distribution.
        """
        self.distribution = distribution
        self.scalar_param = scalar_param

        super().__init__(
            index=index if index is not None else distribution.index,
            columns=columns if columns is not None else distribution.columns,
        )

    def _mean(self):
        """Mean of the distribution - just delegates to inner distribution."""
        return self.distribution.mean()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        from skpro.distributions import Normal

        params1 = {
            "distribution": Normal(mu=[[1, 2], [3, 4]], sigma=1),
            "scalar_param": 2.0,
        }
        params2 = {
            "distribution": Normal(mu=0, sigma=1),
            "scalar_param": 1.0,
        }
        return [params1, params2]


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_subset_param_with_distribution_object():
    """Test that _subset_param handles BaseDistribution objects correctly."""
    from skpro.distributions import Normal

    inner_dist = Normal(
        mu=[[1, 2, 3], [4, 5, 6]],
        sigma=1,
        index=pd.Index([0, 1]),
        columns=pd.Index(["a", "b", "c"]),
    )
    composite = _CompositeDistributionTester(distribution=inner_dist)
    subset_iloc = composite._subset_param(
        inner_dist, rowidx=[0], colidx=None, coerce_scalar=False
    )

    assert isinstance(subset_iloc, BaseDistribution)
    assert subset_iloc.shape == (1, 3)

    expected_means = np.array([[1, 2, 3]])
    np.testing.assert_array_almost_equal(subset_iloc.mean().values, expected_means)
    subset_iat = composite._subset_param(
        inner_dist, rowidx=0, colidx=0, coerce_scalar=True
    )

    assert isinstance(subset_iat, BaseDistribution)
    assert subset_iat.ndim == 0
    assert subset_iat.mean() == 1


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_composite_distribution_iloc_iat():
    """Test that composite distributions work with default iloc/iat."""
    from skpro.distributions import Normal

    inner_dist = Normal(
        mu=[[1, 2], [3, 4], [5, 6]],
        sigma=1,
        index=pd.Index([0, 1, 2]),
        columns=pd.Index(["x", "y"]),
    )

    composite = _CompositeDistributionTester(distribution=inner_dist, scalar_param=10)
    subset = composite.iloc[[0, 1], :]

    assert isinstance(subset, _CompositeDistributionTester)
    assert subset.shape == (2, 2)
    assert subset.scalar_param == 10  # Scalar param should be preserved
    assert subset.distribution.shape == (2, 2)

    expected_means = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_almost_equal(
        subset.distribution.mean().values, expected_means
    )
    subset_col = composite.iloc[:, [0]]

    assert subset_col.shape == (3, 1)
    assert subset_col.distribution.shape == (3, 1)

    scalar = composite.iat[1, 1]

    assert isinstance(scalar, _CompositeDistributionTester)
    assert scalar.ndim == 0
    assert scalar.distribution.ndim == 0
    assert scalar.distribution.mean() == 4


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_subset_param_with_none_indices():
    """Test _subset_param with None indices (no subsetting)."""
    from skpro.distributions import Normal

    inner_dist = Normal(
        mu=[[1, 2], [3, 4]],
        sigma=1,
        index=pd.Index([0, 1]),
        columns=pd.Index(["a", "b"]),
    )

    composite = _CompositeDistributionTester(distribution=inner_dist)
    result = composite._subset_param(
        inner_dist, rowidx=None, colidx=None, coerce_scalar=False
    )

    assert isinstance(result, BaseDistribution)
    assert result.shape == inner_dist.shape
    np.testing.assert_array_equal(result.mean().values, inner_dist.mean().values)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_subset_param_backward_compatibility():
    """Test that _subset_param still works correctly with array-like parameters."""
    from skpro.distributions import Normal

    dist = Normal(mu=[[1, 2], [3, 4]], sigma=1)
    mu_array = np.array([[1, 2], [3, 4]])
    subset_mu = dist._subset_param(
        mu_array, rowidx=[0], colidx=None, coerce_scalar=False
    )

    assert isinstance(subset_mu, np.ndarray)
    np.testing.assert_array_equal(subset_mu, np.array([[1, 2]]))

    scalar_mu = dist._subset_param(mu_array, rowidx=0, colidx=0, coerce_scalar=True)

    assert isinstance(scalar_mu, (np.ndarray, float, int))
    assert scalar_mu == 1


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_composite_distribution_with_scalar_inner():
    """Test composite distribution with scalar inner distribution."""
    from skpro.distributions import Normal

    inner_dist = Normal(mu=5, sigma=2)
    composite = _CompositeDistributionTester(distribution=inner_dist, scalar_param=3)

    assert composite.ndim == 0
    assert composite.distribution.ndim == 0

    scalar_result = composite.iat[0, 0]

    assert isinstance(scalar_result, _CompositeDistributionTester)
    assert scalar_result.ndim == 0
    assert scalar_result.distribution.mean() == 5
