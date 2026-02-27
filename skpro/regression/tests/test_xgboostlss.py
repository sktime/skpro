"""Tests for the XGBoostLSS regressor."""

import numpy as np
import pandas as pd
import pytest

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
def test_zinb_parameter_conversion():
    """Test ZINB parameter conversion from xgboostlss to skpro.

    This test ensures that the parameter conversion from xgboostlss
    (PyTorch convention) to skpro correctly computes the mean (mu).

    In PyTorch NegativeBinomial:
        mean = total_count * probs / (1 - probs)

    where probs is the probability of SUCCESS.

    This is a regression test for a bug where the formula was inverted
    to total_count * (1 - probs) / probs, causing severe under-coverage
    when probs > 0.5.
    """
    # Create a mock XGBoostLSS instance to access the conversion method
    reg = XGBoostLSS(dist="ZINB", n_trials=0)

    # Simulate xgboostlss predicted parameters (PyTorch convention)
    test_cases = [
        {"total_count": 5.0, "probs": 0.3, "gate": 0.2},
        {"total_count": 5.0, "probs": 0.5, "gate": 0.1},
        {"total_count": 5.0, "probs": 0.7, "gate": 0.2},
        {"total_count": 10.0, "probs": 0.8, "gate": 0.15},
    ]

    for case in test_cases:
        # Create DataFrame mimicking xgboostlss output columns
        df = pd.DataFrame(
            {
                "total_count": [case["total_count"]],
                "probs": [case["probs"]],
                "gate": [case["gate"]],
            }
        )

        # Get converted parameters
        skpro_params = reg._get_skpro_val_dict("ZINB", df)

        # Expected mu using PyTorch convention: total_count * probs / (1 - probs)
        expected_mu = case["total_count"] * case["probs"] / (1 - case["probs"])
        expected_alpha = case["total_count"]
        expected_pi = case["gate"]

        actual_mu = float(skpro_params["mu"].squeeze())
        actual_alpha = float(skpro_params["alpha"].squeeze())
        actual_pi = float(skpro_params["pi"].squeeze())

        np.testing.assert_allclose(
            actual_mu,
            expected_mu,
            rtol=1e-6,
            err_msg=f"mu mismatch for probs={case['probs']}: "
            f"got {actual_mu}, expected {expected_mu}",
        )
        np.testing.assert_allclose(actual_alpha, expected_alpha, rtol=1e-6)
        np.testing.assert_allclose(actual_pi, expected_pi, rtol=1e-6)


@pytest.mark.skipif(
    not run_test_for_class(XGBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_zinb_parameter_conversion_matches_pytorch():
    """Test that ZINB parameter conversion matches PyTorch NegativeBinomial.

    This test verifies the conversion by comparing skpro ZINB mean
    with PyTorch NegativeBinomial mean.
    """
    try:
        import torch
        from torch.distributions import NegativeBinomial
    except ImportError:
        pytest.skip("PyTorch not available")

    from skpro.distributions import ZINB

    reg = XGBoostLSS(dist="ZINB", n_trials=0)

    test_cases = [
        {"total_count": 5.0, "probs": 0.3, "gate": 0.0},  # No zero-inflation
        {"total_count": 5.0, "probs": 0.7, "gate": 0.0},
        {"total_count": 10.0, "probs": 0.8, "gate": 0.0},
    ]

    for case in test_cases:
        df = pd.DataFrame(
            {
                "total_count": [case["total_count"]],
                "probs": [case["probs"]],
                "gate": [case["gate"]],
            }
        )

        # Get skpro ZINB parameters
        skpro_params = reg._get_skpro_val_dict("ZINB", df)
        skpro_zinb = ZINB(
            mu=float(skpro_params["mu"].squeeze()),
            alpha=float(skpro_params["alpha"].squeeze()),
            pi=float(skpro_params["pi"].squeeze()),
        )
        skpro_mean = float(np.array(skpro_zinb.mean()).flat[0])

        # PyTorch NegativeBinomial mean (without zero-inflation for pi=0)
        pytorch_nb = NegativeBinomial(
            total_count=torch.tensor(case["total_count"]),
            probs=torch.tensor(case["probs"]),
        )
        pytorch_mean = pytorch_nb.mean.item()

        # For pi=0 (no zero-inflation), ZINB mean should match NB mean
        np.testing.assert_allclose(
            skpro_mean,
            pytorch_mean,
            rtol=1e-4,
            err_msg=f"Mean mismatch for probs={case['probs']}: "
            f"skpro={skpro_mean}, pytorch={pytorch_mean}",
        )
