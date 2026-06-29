"""Tests for residual probabilistic regressors."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import TweedieRegressor

from skpro.regression.residual import ResidualDouble


def _make_positive_data(n=200, p=5, seed=42):
    """Create synthetic positive-response data for Tweedie-style models."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, p)
    beta = rng.uniform(-0.5, 0.8, size=p)
    mu = np.exp(X @ beta)

    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(p)])
    y = pd.Series(mu, name="y")
    return X, y


def test_residual_double_response_lb_clips_quantiles_and_intervals():
    """Quantile/interval outputs should respect configured lower bound."""
    X, y = _make_positive_data()

    reg_mean = TweedieRegressor(power=1.5, link="log")
    reg_resid = DummyRegressor(strategy="constant", constant=3.0)

    reg_unclipped = ResidualDouble(reg_mean, reg_resid, min_scale=1e-6)
    reg_unclipped.fit(X, y)
    q_unclipped = reg_unclipped.predict_quantiles(X, alpha=[0.05, 0.5, 0.95])

    reg_clipped = ResidualDouble(
        reg_mean,
        reg_resid,
        min_scale=1e-6,
        response_lb=0.0,
    )
    reg_clipped.fit(X, y)
    q_clipped = reg_clipped.predict_quantiles(X, alpha=[0.05, 0.5, 0.95])
    i_clipped = reg_clipped.predict_interval(X, coverage=[0.9])

    # test is meaningful: unclipped lower quantiles should contain negatives
    assert (q_unclipped < 0).to_numpy().any()

    # clipping is active and exact for quantiles
    expected = q_unclipped.clip(lower=0.0)
    pd.testing.assert_frame_equal(q_clipped, expected)

    # interval lower/upper bounds must be >= 0 when response_lb=0
    assert (i_clipped >= 0).to_numpy().all()
