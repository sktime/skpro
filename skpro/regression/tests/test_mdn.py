"""Additional tests for MDNRegressor."""

import numpy as np
import pandas as pd
import pytest

from skpro.regression.mdn import MDNRegressor
from skpro.tests.test_switch import run_test_for_class


def test_mdn_loss_name_validation():
    """Loss name validation should accept nll/ngem and reject invalid values."""
    reg = object.__new__(MDNRegressor)

    reg.loss = "nll"
    assert reg._resolve_loss_name() == "nll"

    reg.loss = "ngem"
    assert reg._resolve_loss_name() == "ngem"

    reg.loss = "invalid"
    with pytest.raises(ValueError, match="loss must be 'nll' or 'ngem'"):
        reg._resolve_loss_name()


def test_mdn_ngem_hparams_resolution():
    """Test effective hparam resolution for nGEM path."""
    reg = object.__new__(MDNRegressor)
    reg.lr = 0.01
    reg.ngem_lr_scale = 0.5
    reg.ngem_grad_clip_norm = 1.0

    lr_nll, clip_nll = reg._resolve_training_hparams("nll")
    assert np.isclose(lr_nll, 0.01)
    assert clip_nll is None

    lr_ngem, clip_ngem = reg._resolve_training_hparams("ngem")
    assert np.isclose(lr_ngem, 0.005)
    assert np.isclose(clip_ngem, 1.0)

    reg.ngem_grad_clip_norm = None
    lr_ngem, clip_ngem = reg._resolve_training_hparams("ngem")
    assert np.isclose(lr_ngem, 0.005)
    assert clip_ngem is None


def test_mdn_ngem_hparams_validation():
    """Invalid nGEM training hparams should raise clear errors."""
    reg = object.__new__(MDNRegressor)
    reg.lr = 0.01
    reg.ngem_lr_scale = 0.5
    reg.ngem_grad_clip_norm = 1.0

    reg.lr = 0.0
    with pytest.raises(ValueError, match="lr must be positive"):
        reg._resolve_training_hparams("nll")

    reg.lr = 0.01
    reg.ngem_lr_scale = 0.0
    with pytest.raises(ValueError, match="ngem_lr_scale must be positive"):
        reg._resolve_training_hparams("ngem")

    reg.ngem_lr_scale = 0.5
    reg.ngem_grad_clip_norm = 0.0
    with pytest.raises(ValueError, match="ngem_grad_clip_norm must be positive"):
        reg._resolve_training_hparams("ngem")


def test_mdn_noise_scale_method_values_and_isj():
    """Test noise-scale methods for canonical options including ISJ."""
    reg = object.__new__(MDNRegressor)
    reg.noise_scale_method = "silverman"
    silverman = reg._noise_scale(n_samples=200, total_dim=3)

    reg.noise_scale_method = "scott"
    scott = reg._noise_scale(n_samples=200, total_dim=3)

    reg.noise_scale_method = "constant"
    constant = reg._noise_scale(n_samples=200, total_dim=3)
    assert np.isclose(constant, 1.0)
    assert silverman > 0
    assert scott > 0

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 4))
    y = rng.normal(size=(200, 1))
    reg.noise_scale_method = "isj"
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
def test_mdn_noise_scale_method_performance_comparison():
    """Benchmark-like test comparing MDN performance across noise methods."""
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
            noise_scale_method=schedule,
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


@pytest.mark.skipif(
    not run_test_for_class(MDNRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_mdn_ngem_loss_smoke():
    """Smoke test for nGEM loss training path."""
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X_arr, y_arr = make_regression(
        n_samples=80,
        n_features=4,
        n_informative=3,
        noise=0.1,
        random_state=42,
    )

    X = pd.DataFrame(X_arr)
    y = pd.DataFrame(y_arr, columns=["target"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    reg = MDNRegressor(
        n_gaussians=2,
        hidden_dims=[8],
        epochs=4,
        batch_size=16,
        optimizer="ADAM",
        loss="ngem",
        random_state=42,
    )

    reg.fit(X_train, y_train)
    y_pred = reg.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert len(reg.losses_) == reg.epochs
    assert np.all(np.isfinite(reg.losses_))
