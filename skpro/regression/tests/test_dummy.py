"""Tests for DummyProbaRegressor."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


@pytest.fixture
def diabetes_data():
    """Return a small diabetes dataset split for train/test."""
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.iloc[:50]
    y = pd.DataFrame(y.iloc[:50])
    return train_test_split(X, y, random_state=42)


class TestDummyProbaRegressorPredictVar:
    """Tests that predict_var returns variance (sigma^2), not std dev (sigma)."""

    def test_predict_var_equals_training_variance_empirical(self, diabetes_data):
        """predict_var must equal np.var(y_train) for 'empirical' strategy."""
        from skpro.regression.dummy import DummyProbaRegressor

        X_train, X_test, y_train, _ = diabetes_data

        reg = DummyProbaRegressor(strategy="empirical")
        reg.fit(X_train, y_train)

        y_pred_var = reg.predict_var(X_test)

        expected_var = np.var(y_train.values)

        # Every row of predict_var should equal the training variance
        assert isinstance(y_pred_var, pd.DataFrame)
        assert y_pred_var.shape[0] == X_test.shape[0]
        np.testing.assert_allclose(
            y_pred_var.values,
            expected_var,
            rtol=1e-5,
            err_msg=(
                "predict_var returned std dev instead of variance. "
                f"Got {y_pred_var.values[0, 0]:.6f}, "
                f"expected variance {expected_var:.6f}, "
                f"std dev would be {np.std(y_train.values):.6f}."
            ),
        )

    def test_predict_var_equals_training_variance_normal(self, diabetes_data):
        """predict_var must equal np.var(y_train) for 'normal' strategy."""
        from skpro.regression.dummy import DummyProbaRegressor

        X_train, X_test, y_train, _ = diabetes_data

        reg = DummyProbaRegressor(strategy="normal")
        reg.fit(X_train, y_train)

        y_pred_var = reg.predict_var(X_test)

        expected_var = np.var(y_train.values)

        assert isinstance(y_pred_var, pd.DataFrame)
        np.testing.assert_allclose(
            y_pred_var.values,
            expected_var,
            rtol=1e-5,
            err_msg=(
                "predict_var returned std dev instead of variance. "
                f"Got {y_pred_var.values[0, 0]:.6f}, "
                f"expected variance {expected_var:.6f}."
            ),
        )

    def test_predict_var_not_equal_std(self, diabetes_data):
        """Regression test: predict_var must NOT equal std dev of training labels."""
        from skpro.regression.dummy import DummyProbaRegressor

        X_train, X_test, y_train, _ = diabetes_data

        reg = DummyProbaRegressor(strategy="empirical")
        reg.fit(X_train, y_train)

        y_pred_var = reg.predict_var(X_test)

        std_dev = np.std(y_train.values)
        variance = np.var(y_train.values)

        # std dev and variance differ substantially; the prediction should match var
        assert not np.allclose(y_pred_var.values, std_dev, rtol=1e-3), (
            "predict_var is returning std dev instead of variance. "
            f"std_dev={std_dev:.4f}, variance={variance:.4f}."
        )

    def test_predict_var_is_nonnegative(self, diabetes_data):
        """Variance predictions must be non-negative."""
        from skpro.regression.dummy import DummyProbaRegressor

        X_train, X_test, y_train, _ = diabetes_data

        for strategy in ["empirical", "normal"]:
            reg = DummyProbaRegressor(strategy=strategy)
            reg.fit(X_train, y_train)
            y_pred_var = reg.predict_var(X_test)
            assert (y_pred_var.values >= 0).all(), (
                f"predict_var returned negative values for strategy='{strategy}'"
            )

    def test_predict_var_index_and_columns(self, diabetes_data):
        """predict_var must have same index as X_test and same columns as y_train."""
        from skpro.regression.dummy import DummyProbaRegressor

        X_train, X_test, y_train, _ = diabetes_data

        reg = DummyProbaRegressor(strategy="normal")
        reg.fit(X_train, y_train)

        y_pred_var = reg.predict_var(X_test)

        assert (y_pred_var.index == X_test.index).all()
        assert (y_pred_var.columns == y_train.columns).all()
