"""Test that every distribution in ``__all__`` is importable from the package."""

import pytest

import skpro.distributions as distributions
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_all_names_importable():
    """Every name in ``__all__`` must be importable from ``skpro.distributions``.

    Regression test: ``Cauchy`` was listed in ``__all__`` but its import line was
    missing from ``__init__.py``, so ``from skpro.distributions import Cauchy``
    raised ``ImportError`` while the class existed in ``cauchy.py``.
    """
    missing = [
        name for name in distributions.__all__ if not hasattr(distributions, name)
    ]
    assert not missing, f"names in __all__ but not importable: {missing}"
