"""Tests for the XGBoostLSS regressor."""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import NegativeBinomial
from skpro.regression.xgboostlss import XGBoostLSS
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(XGBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_xgboostlss_params_no_optuna():
    """Test simple use of XGBoostLSS regressor."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg_proba = XGBoostLSS(n_trials=0, max_depth=2)
    reg_proba.fit(X_train, y_train)

    y_pred = reg_proba.predict_proba(X_test)
    trees_df = reg_proba.xgblss_.booster.trees_to_dataframe()
    max_nodes_per_tree = trees_df.groupby("Tree")["Node"].max()
    # All trees should have Node max <= 6 (for max_depth=2)
    assert (max_nodes_per_tree <= 6).all()
    assert y_pred.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(XGBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "params, expected_xgb_params, should_error",
    [
        ({"learning_rate": 0.1}, {"eta": 0.1}, False),
        ({"eta": 0.2}, {"eta": 0.2}, False),
        ({"n_estimators": 50}, {}, False),  # not an xgb_param
        ({"learning_rate": 0.1, "eta": 0.2}, {}, True),
        ({"n_estimators": 50, "num_boost_round": 200}, {}, True),
    ],
)
def test_xgboostlss_param_handling(params, expected_xgb_params, should_error):
    """Test parameter aliases and training params."""
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:50]  # smaller for speed
    y = y.iloc[:50]

    reg = XGBoostLSS(n_trials=0, **params)

    if should_error:
        with pytest.raises(ValueError):
            reg.fit(X, y)
    else:
        reg.fit(X, y)
        y_pred = reg.predict_proba(X)
        assert y_pred.shape == y.shape

        for key, value in expected_xgb_params.items():
            assert reg.xgb_params_.get(key) == value


@pytest.mark.skipif(
    not run_test_for_class(XGBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_xgboostlss_negative_binomial_param_conversion():
    """Test conversion of xgboostlss NegativeBinomial parameters to skpro."""
    import torch

    reg = XGBoostLSS(dist="NegativeBinomial")

    # mean and variance must match the torch distribution fitted by xgboostlss
    params = pd.DataFrame(
        {"total_count": [0.05, 0.8, 3.0, 25.0], "probs": [0.001, 0.2, 0.5, 0.999]}
    )
    skpro_dist = NegativeBinomial(**reg._get_skpro_val_dict("NegativeBinomial", params))
    torch_dist = torch.distributions.NegativeBinomial(
        total_count=torch.tensor(params["total_count"].to_numpy()),
        probs=torch.tensor(params["probs"].to_numpy()),
    )
    np.testing.assert_allclose(skpro_dist.mean().values.ravel(), torch_dist.mean)
    np.testing.assert_allclose(skpro_dist.var().values.ravel(), torch_dist.variance)

    # saturated float32 probabilities must still give a finite mean
    params = pd.DataFrame({"total_count": [2.0], "probs": [1.0]}, dtype="float32")
    skpro_dist = NegativeBinomial(**reg._get_skpro_val_dict("NegativeBinomial", params))
    assert np.isfinite(skpro_dist.mean().values).all()


@pytest.mark.skipif(
    not run_test_for_class(XGBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_xgboostlss_negative_binomial():
    """Test XGBoostLSS with NegativeBinomial distribution on count data."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)))
    y = pd.DataFrame(rng.negative_binomial(3, 3 / (3 + np.exp(1.5 + X[0]))))

    reg = XGBoostLSS(
        dist="NegativeBinomial", response_fn="softplus", n_trials=0, n_cpu=1
    )
    reg.fit(X, y)

    # response_fn applies to total_count, probs always uses sigmoid
    response_fns = reg.xgblss_.dist.param_dict
    assert response_fns["total_count"].__name__ == "softplus_fn"
    assert response_fns["probs"].__name__ == "sigmoid_fn"

    y_pred = reg.predict_proba(X)
    assert isinstance(y_pred, NegativeBinomial)
    quantiles = y_pred.ppf(0.84).values
    np.testing.assert_array_equal(quantiles, np.round(quantiles))
