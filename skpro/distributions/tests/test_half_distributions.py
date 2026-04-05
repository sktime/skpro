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
@pytest.mark.parametrize(
    "dist_class,scipy_func,param_name,param_value,x,p",
    [
        (HalfNormal, halfnorm, "sigma", 2.5, 1.7, 0.8),
        (HalfLogistic, halflogistic, "beta", 1.8, 1.3, 0.7),
    ],
)
def test_half_scalar_matches_scipy_scale(dist_class, scipy_func, param_name, param_value, x, p):
    """Half-distributions should match scipy with scale parameter."""
    dist = dist_class(**{param_name: param_value})

    assert_allclose(
        dist.pdf(x), scipy_func.pdf(x, scale=param_value), rtol=1e-10, atol=1e-14
    )
    assert_allclose(
        dist.cdf(x), scipy_func.cdf(x, scale=param_value), rtol=1e-10, atol=1e-14
    )
    assert_allclose(
        dist.ppf(p), scipy_func.ppf(p, scale=param_value), rtol=1e-10, atol=1e-14
    )


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize(
    "dist_class,scipy_func,param_name,param_array",
    [
        (HalfNormal, halfnorm, "sigma", [[1.0, 2.0], [3.0, 4.0]]),
        (HalfLogistic, halflogistic, "beta", [[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_half_broadcast_and_shape(dist_class, scipy_func, param_name, param_array):
    """Half-distributions should preserve broadcasted shape/index/columns."""
    dist = dist_class(**{param_name: param_array})

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

    param_np = np.asarray(param_array)
    assert_allclose(
        pdf.to_numpy(), scipy_func.pdf(x.to_numpy(), scale=param_np), rtol=1e-10, atol=1e-14
    )
    assert_allclose(
        cdf.to_numpy(), scipy_func.cdf(x.to_numpy(), scale=param_np), rtol=1e-10, atol=1e-14
    )
    assert_allclose(
        ppf.to_numpy(), scipy_func.ppf(p.to_numpy(), scale=param_np), rtol=1e-10, atol=1e-14
    )
