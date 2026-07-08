# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Comprehensive tests for docstring formula injection architecture."""

import pytest

from skpro.distributions.exponential import Exponential
from skpro.distributions.laplace import Laplace
from skpro.distributions.normal import Normal
from skpro.distributions.rayleigh import Rayleigh
from skpro.distributions.weibull import Weibull

# 1. The methods we modified in BaseDistribution
TARGET_METHODS = [
    "pdf",
    "log_pdf",
    "cdf",
    "pmf",
    "log_pmf",
    "ppf",
    "surv",
    "haz",
    "energy",
    "mean",
    "var",
    "pdfnorm",
]

# 2. Classes where we implemented formula hooks
HOOKED_CLASSES = [Normal, Rayleigh, Exponential, Laplace]

# 3. Classes where we DID NOT implement hooks (Control group)
UNHOOKED_CLASSES = [Weibull]


@pytest.mark.parametrize("dist_cls", HOOKED_CLASSES + UNHOOKED_CLASSES)
@pytest.mark.parametrize("method_name", TARGET_METHODS)
def test_placeholder_removal_universal(dist_cls, method_name):
    """Verify {formula_doc} is NEVER visible to users in any distribution."""
    method = getattr(dist_cls, method_name)
    doc = method.__doc__
    assert "{formula_doc}" not in doc, f"Leak in {dist_cls.__name__}.{method_name}"


@pytest.mark.parametrize("dist_cls", HOOKED_CLASSES)
def test_hooked_math_injection(dist_cls):
    """Verify that distributions with hooks actually show LaTeX math."""
    # We check the primary method (pdf)
    doc = dist_cls.pdf.__doc__

    assert ".. math::" in doc, f"Math missing in {dist_cls.__name__}.pdf"
    assert "f(x" in doc, f"Formula content missing in {dist_cls.__name__}.pdf"


@pytest.mark.parametrize("dist_cls", UNHOOKED_CLASSES)
@pytest.mark.parametrize("method_name", ["pdf", "mean", "energy"])
def test_unhooked_clean_fallback(dist_cls, method_name):
    """Verify that unhooked classes remain generic and clean."""
    doc = getattr(dist_cls, method_name).__doc__
    # Should not have math block
    assert (
        ".. math::" not in doc
    ), f"Ghost math block in {dist_cls.__name__}.{method_name}"
    # Should have the original generic preamble
    assert "with the distribution of" in doc


def test_wrapper_execution_safety():
    """Verify wrapping doesn't break execution logic."""
    # Using Rayleigh as the test subject
    dist = Rayleigh.create_test_instance()
    try:
        dist.pdf(1.0)
        dist.mean()
        dist.energy()
    except Exception as e:
        pytest.fail(f"Architecture broke execution of Rayleigh: {e}")


def test_metadata_integrity_full_check():
    """Verify that functools.wraps works for all 12 methods on Rayleigh."""
    for method_name in TARGET_METHODS:
        method = getattr(Rayleigh, method_name)
        assert method.__name__ == method_name
        # Check that it points to the correct base module
        assert "skpro.distributions.base" in method.__module__
