"""Regression tests for half-distribution scipy parameter mappings."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import halflogistic, halfnorm

from skpro.distributions.halflogistic import HalfLogistic
from skpro.distributions.halfnormal import HalfNormal
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_halfnormal_scalar_matches_scipy_scale():
    """HalfNormal scalar pdf/cdf/ppf should match scipy with scale=sigma."""
    sigma = 2.5
    dist = HalfNormal(sigma=sigma)

    x = 1.7
    p = 0.8

    assert_allclose(dist.pdf(x), halfnorm.pdf(x, scale=sigma), rtol=1e-12)
    assert_allclose(dist.cdf(x), halfnorm.cdf(x, scale=sigma), rtol=1e-12)
    assert_allclose(dist.ppf(p), halfnorm.ppf(p, scale=sigma), rtol=1e-12)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_halflogistic_scalar_matches_scipy_scale():
    """HalfLogistic scalar pdf/cdf/ppf should match scipy with scale=beta."""
    beta = 1.8
    dist = HalfLogistic(beta=beta)

    x = 1.3
    p = 0.7

    assert_allclose(dist.pdf(x), halflogistic.pdf(x, scale=beta), rtol=1e-12)
    assert_allclose(dist.cdf(x), halflogistic.cdf(x, scale=beta), rtol=1e-12)
    assert_allclose(dist.ppf(p), halflogistic.ppf(p, scale=beta), rtol=1e-12)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_halfnormal_broadcast_and_shape():
    """HalfNormal should preserve broadcasted shape/index/columns."""
    sigma = [[1.0, 2.0], [3.0, 4.0]]
    dist = HalfNormal(sigma=sigma)

    x = pd.DataFrame([[0.5, 1.0], [2.0, 3.0]], index=dist.index, columns=dist.columns)
    p = pd.DataFrame([[0.1, 0.2], [0.7, 0.9]], index=dist.index, columns=dist.columns)

    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    ppf = dist.ppf(p)

    assert pdf.shape == dist.shape
    assert cdf.shape == dist.shape
    assert ppf.shape == dist.shape
    assert pdf.index.equals(dist.index)
    assert cdf.index.equals(dist.index)
    assert ppf.index.equals(dist.index)
    assert pdf.columns.equals(dist.columns)
    assert cdf.columns.equals(dist.columns)
    assert ppf.columns.equals(dist.columns)

    sigma_np = np.asarray(sigma)
    assert_allclose(pdf.to_numpy(), halfnorm.pdf(x.to_numpy(), scale=sigma_np), rtol=1e-12)
    assert_allclose(cdf.to_numpy(), halfnorm.cdf(x.to_numpy(), scale=sigma_np), rtol=1e-12)
    assert_allclose(ppf.to_numpy(), halfnorm.ppf(p.to_numpy(), scale=sigma_np), rtol=1e-12)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_halflogistic_broadcast_and_shape():
    """HalfLogistic should preserve broadcasted shape/index/columns."""
    beta = [[1.0, 2.0], [3.0, 4.0]]
    dist = HalfLogistic(beta=beta)

    x = pd.DataFrame([[0.5, 1.0], [2.0, 3.0]], index=dist.index, columns=dist.columns)
    p = pd.DataFrame([[0.1, 0.2], [0.7, 0.9]], index=dist.index, columns=dist.columns)

    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    ppf = dist.ppf(p)

    assert pdf.shape == dist.shape
    assert cdf.shape == dist.shape
    assert ppf.shape == dist.shape
    assert pdf.index.equals(dist.index)
    assert cdf.index.equals(dist.index)
    assert ppf.index.equals(dist.index)
    assert pdf.columns.equals(dist.columns)
    assert cdf.columns.equals(dist.columns)
    assert ppf.columns.equals(dist.columns)

    beta_np = np.asarray(beta)
    assert_allclose(pdf.to_numpy(), halflogistic.pdf(x.to_numpy(), scale=beta_np), rtol=1e-12)
    assert_allclose(cdf.to_numpy(), halflogistic.cdf(x.to_numpy(), scale=beta_np), rtol=1e-12)
    assert_allclose(ppf.to_numpy(), halflogistic.ppf(p.to_numpy(), scale=beta_np), rtol=1e-12)
