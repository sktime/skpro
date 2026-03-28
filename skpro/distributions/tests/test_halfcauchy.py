"""Tests for HalfCauchy distribution."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import halfcauchy

from skpro.distributions.halfcauchy import HalfCauchy
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_halfcauchy_scalar_matches_scipy():
    """HalfCauchy scalar pdf/cdf/ppf should match scipy implementation."""
    beta = 2.5
    dist = HalfCauchy(beta=beta)

    x = 1.7
    p = 0.8

    assert_allclose(dist.pdf(x), halfcauchy.pdf(x, scale=beta), rtol=1e-12)
    assert_allclose(dist.cdf(x), halfcauchy.cdf(x, scale=beta), rtol=1e-12)
    assert_allclose(dist.ppf(p), halfcauchy.ppf(p, scale=beta), rtol=1e-12)

    # Check support behavior (x < 0 outside support)
    assert_allclose(dist.pdf(-1.0), 0.0, atol=1e-15)
    assert_allclose(dist.cdf(-1.0), 0.0, atol=1e-15)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_halfcauchy_broadcast_and_shape():
    """HalfCauchy should broadcast and preserve shape/index/columns."""
    beta = [[1.0, 2.0], [3.0, 4.0]]
    dist = HalfCauchy(beta=beta)

    x = pd.DataFrame([[0.5, 1.0], [2.0, 3.0]], index=dist.index, columns=dist.columns)
    p = pd.DataFrame([[0.1, 0.2], [0.7, 0.9]], index=dist.index, columns=dist.columns)

    pdf = dist.pdf(x)
    cdf = dist.cdf(x)
    ppf = dist.ppf(p)

    assert isinstance(pdf, pd.DataFrame)
    assert isinstance(cdf, pd.DataFrame)
    assert isinstance(ppf, pd.DataFrame)

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

    assert_allclose(pdf.to_numpy(), halfcauchy.pdf(x.to_numpy(), scale=beta_np), rtol=1e-12)
    assert_allclose(cdf.to_numpy(), halfcauchy.cdf(x.to_numpy(), scale=beta_np), rtol=1e-12)
    assert_allclose(ppf.to_numpy(), halfcauchy.ppf(p.to_numpy(), scale=beta_np), rtol=1e-12)
