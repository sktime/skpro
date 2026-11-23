"""Tests for ZeroInflated distribution."""

import numpy as np
import pytest

from skpro.distributions import Normal, Poisson, ZeroInflated
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_zeroinflated_capabilities_continuous():
    """Test capability tags when ZeroInflated wraps continuous distribution (#1243)."""
    norm = Normal(mu=1.0, sigma=1.0)
    zi = ZeroInflated(p=0.5, distribution=norm)

    exact = zi.get_tag("capabilities:exact")
    approx = zi.get_tag("capabilities:approx")

    # Continuous base has pdf/log_pdf, no pmf/log_pmf
    assert (
        "pmf" not in exact
    ), "pmf should not be in exact capabilities for continuous base"
    assert (
        "log_pmf" not in exact
    ), "log_pmf should not be in exact capabilities for continuous base"
    assert "pdf" in exact, "pdf should be in exact capabilities"
    assert "log_pdf" in exact, "log_pdf should be in exact capabilities"

    assert "pmf" in approx, "pmf should be moved to approx capabilities"
    assert "log_pmf" in approx, "log_pmf should be moved to approx capabilities"


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_zeroinflated_capabilities_discrete():
    """Test capability tags when ZeroInflated wraps discrete distribution (#1243)."""
    poisson = Poisson(mu=2.0)
    zi = ZeroInflated(p=0.7, distribution=poisson)

    exact = zi.get_tag("capabilities:exact")
    approx = zi.get_tag("capabilities:approx")

    assert "pmf" in exact, "pmf should be in exact capabilities"
    assert "log_pmf" in exact, "log_pmf should be in exact capabilities"
    assert "pdf" not in exact, "pdf should not be in exact capabilities"
    assert "log_pdf" not in exact, "log_pdf should not be in exact capabilities"

    assert "pdf" in approx, "pdf should be moved to approx capabilities"
    assert "log_pdf" in approx, "log_pdf should be moved to approx capabilities"


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize("params", ZeroInflated.get_test_params())
def test_zeroinflated_less_than_zero(params):
    """Test boundary behavior for negative values."""
    zi = ZeroInflated(**params)

    v = -1.0
    funcs_and_expected = [
        (zi.cdf, 0.0),
        (zi.pdf, 0.0),
        (zi.pmf, 0.0),
        (zi.log_pdf, -np.inf),
        (zi.log_pmf, -np.inf),
    ]

    for func, expected in funcs_and_expected:
        values = func(v)
        assert (np.asarray(values) == expected).all()
