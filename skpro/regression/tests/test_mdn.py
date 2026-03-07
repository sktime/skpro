"""Additional tests for MDNRegressor."""

import numpy as np
import pandas as pd
import pytest

from skpro.regression.mdn import MDNRegressor
from skpro.tests.test_switch import run_test_for_class


def test_mdn_noise_schedule_values_and_isj():
    """Test schedule scaling for canonical options including ISJ."""
    reg = object.__new__(MDNRegressor)
    reg.noise_schedule = "silverman"
    silverman = reg._noise_scale(n_samples=200, total_dim=3)

    reg.noise_schedule = "scott"
    scott = reg._noise_scale(n_samples=200, total_dim=3)

    reg.noise_schedule = "constant"
    constant = reg._noise_scale(n_samples=200, total_dim=3)
    assert np.isclose(constant, 1.0)
    assert silverman > 0
    assert scott > 0

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 4))
    y = rng.normal(size=(200, 1))
    reg.noise_schedule = "isj"
    isj = reg._noise_scale(n_samples=200, total_dim=5, X=X, y=y)
    assert np.isfinite(isj)
    assert isj > 0


@pytest.mark.skipif(
    not run_test_for_class(MDNRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_mdn_custom_activation_and_optimizer_class():
    """Test MDN supports custom hidden activation and optimizer class."""
    import torch.nn as nn
    from pytorch_optimizer import AdamP
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X_arr, y_arr = make_regression(
        n_samples=60,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42,
    )

    X = pd.DataFrame(X_arr)
    y = pd.DataFrame(y_arr, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    reg = MDNRegressor(
        n_gaussians=2,
        hidden_dims=[12, 8],
        activation=nn.SELU,
        optimizer=AdamP,
        epochs=3,
        batch_size=16,
        random_state=42,
    )

    reg.fit(X_train, y_train)
    y_pred = reg.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert y_pred.mean().shape == y_test.shape
    assert any(isinstance(layer, nn.SELU) for layer in reg.model_.backbone)


@pytest.mark.skipif(
    not run_test_for_class(MDNRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_mdn_noise_schedule_performance_comparison():
    """Benchmark-like test comparing MDN performance across noise schedules."""
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X_arr, y_arr = make_regression(
        n_samples=1000,
        n_features=5,
        n_informative=4,
        noise=8.0,
        random_state=42,
    )

    X = pd.DataFrame(X_arr)
    y = pd.DataFrame(y_arr, columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    schedules = ["constant", "scott", "silverman", "isj"]
    nll_by_schedule = {}
    for schedule in schedules:
        reg = MDNRegressor(
            n_gaussians=3,
            hidden_dims=[16, 8],
            epochs=100,
            lr=0.01,
            batch_size=32,
            input_noise_std=0.05,
            target_noise_std=0.02,
            noise_schedule=schedule,
            optimizer="ADAM",
            random_state=42,
        )

        reg.fit(X_train, y_train)
        y_pred = reg.predict_proba(X_test)
        log_pdf = y_pred.log_pdf(y_test)
        nll = -float(np.asarray(log_pdf).mean())
        nll_by_schedule[schedule] = nll

    for schedule, nll in nll_by_schedule.items():
        assert np.isfinite(nll), f"NLL must be finite for schedule '{schedule}'"

    unique_scores = {round(v, 8) for v in nll_by_schedule.values()}
    assert len(unique_scores) > 1
