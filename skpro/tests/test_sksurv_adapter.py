"""Tests for _SksurvAdapter."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from skpro.survival.adapters.sksurv import _SksurvAdapter


class MockSksurvAdapter(_SksurvAdapter):
    """Mock adapter for reproducing sksurv adapter behavior."""

    def _get_sksurv_class(self):
        return MagicMock()

    def get_params(self, deep=True):
        return {}


def test_sksurv_adapter_probability_mass_and_alignment():
    """Test _SksurvAdapter predict_proba mass and temporal alignment."""
    # Define mock survival results
    mock_surv = np.array([[0.8, 0.5, 0.5]])
    mock_times = np.array([10.0, 20.0, 30.0])

    X = pd.DataFrame({"feature1": [1.0]})

    # Initialize and mock the estimator internals
    adapter = MockSksurvAdapter()
    adapter._estimator = MagicMock()
    adapter._estimator.predict_survival_function = MagicMock(return_value=mock_surv)
    adapter._estimator.unique_times_ = mock_times
    adapter._y_cols = ["time"]

    # Run the predict_proba logic
    dist = adapter._predict_proba(X)

    # Retrieve expected features
    times = dist.spl.values.flatten()
    weights = dist.weights.values

    # Check alignment - Times should not be truncated
    np.testing.assert_allclose(times, mock_times)

    # Expected masses:
    # 1.0 -> 0.8  at t=10.0 (mass 0.2)
    # 0.8 -> 0.5  at t=20.0 (mass 0.3)
    # 0.5 -> 0.5  at t=30.0 (mass 0.0 + 0.5 tail mass = 0.5)
    expected_weights = np.array([0.2, 0.3, 0.5])
    
    np.testing.assert_allclose(weights, expected_weights, atol=1e-7)

    # Verify that total mass is precisely 1.0
    total_mass = weights.sum()
    assert np.isclose(total_mass, 1.0, atol=1e-7), f"Expected total mass 1.0, got {total_mass}"
