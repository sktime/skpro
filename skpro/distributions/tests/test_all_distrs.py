# -*- coding: utf-8 -*-
"""Tests for BaseDistribution API points."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
import pytest
from skbase.testing import BaseFixtureGenerator, QuickTester

from skpro.datatypes import check_is_mtype
from skpro.distributions.base import BaseDistribution
from skpro.tests.test_all_estimators import PackageConfig


class DistributionFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for probability distributions.

    Fixtures parameterized
    ----------------------
    object_class: object inheriting from BaseObject
        ranges over object classes not excluded by EXCLUDE_OBJECTS, EXCLUDED_TESTS
    object_instance: instance of object inheriting from BaseObject
        ranges over object classes not excluded by EXCLUDE_OBJECTS, EXCLUDED_TESTS
        instances are generated by create_test_instance class method
    """

    object_type_filter = BaseDistribution


def _has_capability(distr, method):
    """Check whether distr has capability of method.

    Parameters
    ----------
    distr : BaseDistribution object
    method : str
        method name to check

    Returns
    -------
    whether distr has capability method, according to tags
    capabilities:approx and capabilities:exact
    """
    approx_methods = distr.get_tag("capabilities:approx")
    exact_methods = distr.get_tag("capabilities:exact")
    return method in approx_methods or method in exact_methods


METHODS_SCALAR = ["mean", "var", "energy"]
METHODS_SCALAR_POS = ["var", "energy"]  # result always non-negative?
METHODS_X = ["energy", "pdf", "log_pdf", "cdf"]
METHODS_X_POS = ["energy", "pdf", "cdf"]  # result always non-negative?
METHODS_P = ["ppf"]
METHODS_ROWWISE = ["energy"]  # results in one column


class TestAllDistributions(PackageConfig, DistributionFixtureGenerator, QuickTester):
    """Module level tests for all sktime parameter fitters."""

    @pytest.mark.parametrize("shuffled", [False, True])
    def test_sample(self, object_instance, shuffled):
        """Test sample expected return."""
        d = object_instance

        if shuffled:
            d = _shuffle_distr(d)

        res = d.sample()

        assert d.shape == res.shape
        assert (res.index == d.index).all()
        assert (res.columns == d.columns).all()

        res_panel = d.sample(3)
        dummy_panel = pd.concat([res, res, res], keys=range(3))
        assert dummy_panel.shape == res_panel.shape
        assert (res_panel.index == dummy_panel.index).all()
        assert (res_panel.columns == dummy_panel.columns).all()

    @pytest.mark.parametrize("shuffled", [False, True])
    @pytest.mark.parametrize("method", METHODS_SCALAR, ids=METHODS_SCALAR)
    def test_methods_scalar(self, object_instance, method, shuffled):
        """Test expected return of scalar methods."""
        if not _has_capability(object_instance, method):
            return None

        d = object_instance
        if shuffled:
            d = _shuffle_distr(d)

        res = getattr(object_instance, method)()

        _check_output_format(res, d, method)

    @pytest.mark.parametrize("shuffled", [False, True])
    @pytest.mark.parametrize("method", METHODS_X, ids=METHODS_X)
    def test_methods_x(self, object_instance, method, shuffled):
        """Test expected return of methods that take sample-like argument."""
        if not _has_capability(object_instance, method):
            return None

        d = object_instance

        if shuffled:
            d = _shuffle_distr(d)

        x = d.sample()
        res = getattr(object_instance, method)(x)

        _check_output_format(res, d, method)

    @pytest.mark.parametrize("shuffled", [False, True])
    @pytest.mark.parametrize("method", METHODS_P, ids=METHODS_P)
    def test_methods_p(self, object_instance, method, shuffled):
        """Test expected return of methods that take percentage-like argument."""
        if not _has_capability(object_instance, method):
            return None

        d = object_instance

        if shuffled:
            d = _shuffle_distr(d)

        np_unif = np.random.uniform(size=d.shape)
        p = pd.DataFrame(np_unif, index=d.index, columns=d.columns)
        res = getattr(object_instance, method)(p)

        _check_output_format(res, d, method)

    @pytest.mark.parametrize("q", [0.7, [0.1, 0.3, 0.9]])
    def test_quantile(self, object_instance, q):
        """Test expected return of quantile method."""
        if not _has_capability(object_instance, "ppf"):
            return None

        d = object_instance

        def _check_quantile_output(obj, q):
            assert check_is_mtype(obj, "pred_quantiles", "Proba")
            assert (obj.index == d.index).all()

            if not isinstance(q, list):
                q = [q]
            expected_columns = pd.MultiIndex.from_product([d.columns, q])
            assert (obj.columns == expected_columns).all()

        res = d.quantile(q)
        _check_quantile_output(res, q)


def _check_output_format(res, dist, method):
    """Check output format expectations for BaseDistribution tests."""
    if method in METHODS_ROWWISE:
        exp_shape = (dist.shape[0], 1)
    else:
        exp_shape = dist.shape
    assert res.shape == exp_shape
    assert (res.index == dist.index).all()
    if method not in METHODS_ROWWISE:
        assert (res.columns == dist.columns).all()

    if method in METHODS_SCALAR_POS or method in METHODS_X_POS:
        assert (res >= 0).all().all()


def _shuffle_distr(d):
    """Shuffle distribution row index."""
    shuffled_index = pd.DataFrame(d.index).sample(frac=1).index
    return d.loc[shuffled_index]
