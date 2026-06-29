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
        X = pd.DataFrame(np.random.RandomState(42).randn(50, 1))

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
        """Test that the returned distribution supports mean() and var()."""
        X = pd.DataFrame(np.random.RandomState(42).randn(50, 1))

        fitter = object_instance
        fitter.fit(X)

        dist = fitter.proba()

        mean_val = dist.mean()
        var_val = dist.var()

        assert np.isfinite(mean_val), f"mean() returned non-finite value: {mean_val}"
        assert np.isfinite(var_val), f"var() returned non-finite value: {var_val}"
        assert var_val >= 0, f"var() returned negative value: {var_val}"
