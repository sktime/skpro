"""Tests for the XGBoostLSS regressor."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

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


@pytest.fixture(scope="module", params=["exp", "softplus", "relu"])
def fitted_negative_binomial(request):
    """Fit a negative binomial regressor to reproducible count data."""
    pytest.importorskip("xgboostlss")
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(300, 3)),
        columns=["a", "b", "c"],
        index=pd.Index(range(100, 400), name="sample"),
    )
    mean = np.exp(1.5 + 0.5 * X["a"].to_numpy())
    y = pd.DataFrame({"qty": rng.negative_binomial(3, 3 / (3 + mean))}, index=X.index)
    reg = XGBoostLSS(
        dist="NegativeBinomial",
        response_fn=request.param,
        n_trials=0,
        num_boost_round=20,
        max_depth=2,
        n_cpu=1,
        random_state=0,
    )
    # Keep small fits fast and avoid OpenMP oversubscription with XGBoost.
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield reg.fit(X, y), X
    finally:
        torch.set_num_threads(n_threads)


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboostlss", severity="none"),
    reason="skip test if xgboostlss is not installed in environment",
)
def test_negative_binomial_moments_match_xgboostlss_distribution():
    """Conversion preserves the mean and variance, including small total_count."""
    import torch

    params = pd.DataFrame(
        {
            "total_count": [0.05, 0.2, 0.8, 3.0, 25.0],
            "probs": [0.001, 0.2, 0.5, 0.9, 0.999],
        }
    )
    reg = XGBoostLSS(dist="NegativeBinomial", n_trials=0)
    values = reg._get_skpro_val_dict("NegativeBinomial", params)
    skpro_dist = NegativeBinomial(**values)
    torch_dist = torch.distributions.NegativeBinomial(
        total_count=torch.tensor(params["total_count"].to_numpy()),
        probs=torch.tensor(params["probs"].to_numpy()),
    )

    np.testing.assert_allclose(
        skpro_dist.mean().to_numpy().ravel(), torch_dist.mean.numpy()
    )
    np.testing.assert_allclose(
        skpro_dist.var().to_numpy().ravel(), torch_dist.variance.numpy()
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboostlss", severity="none"),
    reason="skip test if xgboostlss is not installed in environment",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_negative_binomial_saturated_probs_give_finite_moments(dtype):
    """Saturated probabilities give finite moments for float32 and float64."""
    params = pd.DataFrame({"total_count": [0.2, 2.0], "probs": [0.0, 1.0]}, dtype=dtype)
    reg = XGBoostLSS(dist="NegativeBinomial", n_trials=0)
    values = reg._get_skpro_val_dict("NegativeBinomial", params)
    pred = NegativeBinomial(**values)

    assert np.isfinite(pred.mean().to_numpy()).all()
    assert np.isfinite(pred.var().to_numpy()).all()
    np.testing.assert_array_equal(values["alpha"], params[["total_count"]].to_numpy())


def test_negative_binomial_uses_response_fn_for_total_count(fitted_negative_binomial):
    """The requested response function transforms total_count, not probs."""
    reg, _ = fitted_negative_binomial
    response_fns = reg.xgblss_.dist.param_dict

    assert response_fns["total_count"].__name__ == f"{reg.response_fn}_fn"
    assert response_fns["probs"].__name__ == "sigmoid_fn"


def test_negative_binomial_predict_proba_returns_count_distribution(
    fitted_negative_binomial,
):
    """Predictions are count distributions with the original index and columns."""
    reg, X = fitted_negative_binomial
    X_test = X.iloc[3:8]

    pred = reg.predict_proba(X_test)

    assert isinstance(pred, NegativeBinomial)
    pd.testing.assert_index_equal(pred.index, X_test.index)
    pd.testing.assert_index_equal(pred.columns, pd.Index(["qty"]))
    assert pred.shape == (len(X_test), 1)
    assert np.isfinite(pred.mean().to_numpy()).all()
    assert (pred.mean().to_numpy() > 0).all()
    assert (pred.var().to_numpy() > pred.mean().to_numpy()).all()
    quantiles = pred.ppf(0.84).to_numpy()
    assert np.isfinite(quantiles).all()
    assert (quantiles >= 0).all()
    np.testing.assert_array_equal(quantiles, np.round(quantiles))
