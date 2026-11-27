"""Test Pipeline with no transformers (regressor-only).

Regression test for NameError in Pipeline._transform when the pipeline
contains only a regressor and no transformers, causing the for-loop to
never execute and `Xt` to remain undefined.
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from skpro.regression.compose._pipeline import Pipeline
from skpro.regression.residual import ResidualDouble


@pytest.fixture
def diabetes_data():
    """Return a small train/test split of the diabetes dataset."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.iloc[:50]
    y = y.iloc[:50]
    y = pd.DataFrame(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regressor_only_pipeline(diabetes_data):
    """Return a fitted Pipeline with only a regressor (no transformers)."""
    X_train, _, y_train, _ = diabetes_data
    regressor = ResidualDouble.create_test_instance()
    pipe = Pipeline(steps=[regressor])
    pipe.fit(X_train, y_train)
    return pipe


class TestPipelineNoTransformers:
    """Tests for Pipeline containing only a regressor and no transformers."""

    def test_predict_no_transformers(self, regressor_only_pipeline, diabetes_data):
        """Test that predict works without NameError."""
        _, X_test, _, _ = diabetes_data
        y_pred = regressor_only_pipeline.predict(X_test)
        assert isinstance(y_pred, pd.DataFrame)
        assert len(y_pred) == len(X_test)

    def test_predict_proba_no_transformers(
        self, regressor_only_pipeline, diabetes_data
    ):
        """Test that predict_proba works without NameError."""
        _, X_test, _, _ = diabetes_data
        y_pred_proba = regressor_only_pipeline.predict_proba(X_test)
        # BaseDistribution returned; just assert no error
        assert y_pred_proba is not None
        assert len(y_pred_proba) == len(X_test)

    def test_predict_quantiles_no_transformers(
        self, regressor_only_pipeline, diabetes_data
    ):
        """Test that predict_quantiles works without NameError."""
        _, X_test, _, _ = diabetes_data
        y_pred_q = regressor_only_pipeline.predict_quantiles(X_test, alpha=[0.25, 0.75])
        assert isinstance(y_pred_q, pd.DataFrame)
        assert len(y_pred_q) == len(X_test)
