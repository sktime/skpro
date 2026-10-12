"""Tests for the GeneralizedPareto distribution."""

import warnings

import numpy as np
import pytest

from skpro.distributions.base import BaseDistribution
from skpro.distributions.gen_pareto import GeneralizedPareto
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_gen_pareto_tags_match_implementation():
    """Test that every method tagged exact is overridden, not the base default."""
    assert GeneralizedPareto.get_class_tag("distr:paramtype") == "parametric"

    for capability in GeneralizedPareto.get_class_tag("capabilities:exact"):
        method = f"_{capability}"
        assert getattr(GeneralizedPareto, method) is not getattr(
            BaseDistribution, method
        ), f"{capability} is tagged exact but only the base default exists"


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize("params", GeneralizedPareto.get_test_params())
def test_gen_pareto_log_pdf_is_exact(params):
    """Test that log_pdf is log of pdf and falls back on no approximation."""
    distribution = GeneralizedPareto(**params)
    x = distribution.sample()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        log_pdf = distribution.log_pdf(x)

    assert np.allclose(np.asarray(log_pdf), np.log(np.asarray(distribution.pdf(x))))
