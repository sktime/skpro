"""Tests for harmonized distribution keyword handling."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LinearRegression

from skpro.regression.residual import ResidualDouble
from skpro.regression.xgboostlss import XGBoostLSS


def _make_residual(**kwargs):
    return ResidualDouble(estimator=LinearRegression(), **kwargs)


@pytest.mark.parametrize("alias", ["distribution", "dist", "dist_type", "distr_type"])
def test_residual_double_accepts_aliases(alias):
    kwargs = {alias: "Laplace"}
    reg = _make_residual(**kwargs)
    assert reg.distribution == "Laplace"


def test_residual_double_raises_on_conflict():
    with pytest.raises(ValueError, match="conflicting distribution"):
        _make_residual(distribution="Normal", dist="Laplace")


@pytest.mark.parametrize("alias", ["distribution", "dist", "dist_type", "distr_type"])
def test_xgboostlss_accepts_aliases(alias):
    kwargs = {alias: "Gamma"}
    est = XGBoostLSS(**kwargs)
    assert est.distribution == "Gamma"


def test_xgboostlss_raises_on_conflict():
    with pytest.raises(ValueError, match="conflicting distribution"):
        XGBoostLSS(distribution="Normal", dist="Laplace")

