"""Additional tests for MDNRegressor."""

import pandas as pd
import pytest

from skpro.regression.mdn import MDNRegressor
from skpro.tests.test_switch import run_test_for_class


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
