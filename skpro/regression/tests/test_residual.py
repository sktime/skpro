"""Tests Generalized Linear Model regressor."""

from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression

from skpro.regression.residual import ResidualDouble
from skpro.tests.test_switch import run_test_for_class


def held_out_cdf(
    nn: int = 25_000,
    distr_type: Literal["Laplace", "Normal", "t"] = "Laplace",
    model: Literal["linear", "constant"] = "linear",
    trafo: Literal["absolute", "squared"] = "absolute",
    distr_params: Optional[Dict[str, float]] = None,
) -> pd.Series:
    np.random.seed(42)
    if distr_params is None:
        distr_params = {}
    else:
        distr_params = distr_params.copy()
    x_df = pd.DataFrame(
        {"a": np.random.randn(nn), "b": np.random.randn(nn), "c": np.random.randn(nn)}
    ).clip(-2, 2)
    # DGP
    if model == "linear":
        loc_param_vec = pd.Series({"a": -1, "b": 1, "c": 0})
        log_scale_param_vec = pd.Series({"a": 0, "b": 0.01, "c": 0.5})
        loc_vec = x_df.dot(loc_param_vec)
        log_scale_vec = x_df.dot(log_scale_param_vec).round(1)
    else:
        loc_vec = pd.Series(3.0, index=x_df.index)
        log_scale_vec = pd.Series(0.0, index=x_df.index)

    if distr_type == "Laplace":
        dist_cls = stats.laplace
    elif distr_type == "Normal":
        dist_cls = stats.norm
    elif distr_type == "t":
        dist_cls = stats.t
    else:
        raise ValueError(f"Distribution {distr_type} not supported")
    dist = dist_cls(loc=loc_vec, scale=np.exp(log_scale_vec), **distr_params)
    y = pd.DataFrame(dist.rvs((2, nn)).T, index=x_df.index, columns=["y0", "y1"])
    reg = ResidualDouble(
        estimator=LinearRegression(),
        estimator_resid=LinearRegression(),
        distr_params=distr_params,
        distr_type=distr_type,
        residual_trafo=trafo,
        # cv=KFold(n_splits=3),
    )

    reg.fit(x_df, y["y0"])
    pred = reg.predict_proba(x_df)

    cdf = pred.cdf(y[["y1"]])["y0"]
    return cdf


@pytest.mark.skipif(
    not run_test_for_class(ResidualDouble),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "distr_type,distr_params",
    [
        ("t", {"df": 5.1}),
        ("t", {"df": 2.5}),
        ("Laplace", None),
        ("Normal", None),
    ],
)
@pytest.mark.parametrize("trafo", ["absolute", "squared"])
def test_residual_double_constant(distr_type, distr_params, trafo):
    """Test validity of ResidualDouble regressor on a constant model."""
    Q_BINS = 4
    TOL_ALPHA = 0.001
    np.random.seed(42)
    # Should be uniform(0,1)
    held_out_quantiles = held_out_cdf(
        model="constant", distr_type=distr_type, distr_params=distr_params, trafo=trafo
    )
    # Counts of quantiles in bins
    vc = pd.cut(held_out_quantiles, bins=np.linspace(0, 1, Q_BINS + 1)).value_counts()
    # Expected counts under uniformity
    e_vec = vc * vc.sum() / (Q_BINS * vc)
    # Observed counts
    o_vec = vc
    # Chi-squared test
    chsq = stats.chisquare(o_vec, e_vec, ddof=2)
    # dist=1, ddf<3, trafo="squared" does very badly, hence the high tolerance
    assert chsq.pvalue > TOL_ALPHA
