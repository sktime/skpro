"""Automated tests based on the skbase test suite template."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
from skbase.testing import QuickTester

from skpro.distributions.base import BaseDistribution
from skpro.tests.test_all_estimators import BaseFixtureGenerator, PackageConfig


class TestAllDistFitters(PackageConfig, BaseFixtureGenerator, QuickTester):
    """Generic tests for all distribution fitters in skpro."""

    object_type_filter = "distfitter"

    def test_input_output_contract(self, object_instance):
        """Test that fit/proba follow the expected contract."""
        X = pd.DataFrame(np.abs(np.random.RandomState(42).randn(50, 1)) + 0.5)

        fitter = object_instance
        fitter.fit(X)

        dist = fitter.proba()

        assert isinstance(
            dist, BaseDistribution
        ), f"proba() must return a BaseDistribution, got {type(dist)}"
        assert (
            dist.ndim == 0
        ), f"proba() must return a scalar distribution (ndim==0), got ndim={dist.ndim}"

    def test_proba_has_mean_var(self, object_instance):
        """Test that the returned distribution supports mean() and var().

        Some distributions, such as ``Cauchy``, have no mean or variance, and
        declare this through the ``capabilities:undefined`` tag. Only those are
        allowed to return a value that is not a finite number, so a
        distribution that should have a mean, but returns ``inf`` or ``nan``,
        still fails here.
        """
        X = pd.DataFrame(np.abs(np.random.RandomState(42).randn(50, 1)) + 0.5)

        fitter = object_instance
        fitter.fit(X)

        dist = fitter.proba()
        undefined = dist.get_tag("capabilities:undefined", [], raise_error=False)
        undefined = undefined or []

        if "mean" not in undefined:
            mean_val = dist.mean()
            msg = f"mean() returned non-finite value: {mean_val}"
            assert np.isfinite(mean_val), msg

        if "var" not in undefined:
            var_val = dist.var()
            assert np.isfinite(var_val), f"var() returned non-finite value: {var_val}"
            assert var_val >= 0, f"var() returned negative value: {var_val}"
