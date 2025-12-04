"""Tests for DifferentiableTransformer."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from skpro.compose import DifferentiableTransformer
from skpro.regression.compose import TransformedTargetRegressor
from skpro.regression.linear import ARDRegression


@pytest.fixture
def sample_data():
    """Generate sample training and test data as DataFrames."""
    np.random.seed(42)
    size = 1000
    X = pd.DataFrame(np.random.normal(1, 1, (size, 1)), columns=["feature_0"])
    y = pd.DataFrame(
        2 * X.values + np.random.normal(0, 0.5, (size, 1)), columns=["target"]
    )
    return X, y


def test_ttr_pdf_vs_linear_jacobian(sample_data):
    """Compare the TTR PDF with manual transformation with Jacobian."""
    X, y = sample_data
    mms = MinMaxScaler(feature_range=(0.1, 0.9))
    mms_diff = DifferentiableTransformer(transformer=mms)
    est = ARDRegression()

    # full ttr pipeline
    pipe = TransformedTargetRegressor(regressor=est, transformer=mms_diff)
    pipe.fit(X=X, y=y)
    pdf_ttr = pipe.predict_proba(X).pdf(y).values

    # manual transformation
    y_transformed = mms.fit_transform(y)
    est.fit(X=X, y=y_transformed)
    pdf_raw = est.predict_proba(X).pdf(y_transformed).values
    pdf_expected = pdf_raw / mms.scale_

    assert np.allclose(pdf_ttr, pdf_expected, rtol=1e-2)
    assert not np.allclose(pdf_ttr, pdf_raw, rtol=1e-2)


def test_ttr_pdf_vs_nonlinear_jacobian(sample_data):
    """Compare TTR PDF with manual transformation with non-linear Jacobian."""
    X, y = sample_data

    def transform_func_diff(x):
        return expit(x) * (1 - expit(x))

    mms_diff = DifferentiableTransformer(
        transformer=FunctionTransformer(func=expit, inverse_func=logit),
        transform_func_diff=transform_func_diff,
    )

    est = ARDRegression()

    pipe = TransformedTargetRegressor(regressor=est, transformer=mms_diff)
    pipe.fit(X=X, y=y)
    pdf_ttr = pipe.predict_proba(X).pdf(y).values

    y_transformed = expit(y)
    est.fit(X=X, y=y_transformed)
    pdf_raw = est.predict_proba(X).pdf(y_transformed).values
    pdf_expected = pdf_raw / transform_func_diff(y).abs().values

    assert np.allclose(pdf_ttr, pdf_expected, rtol=0.1)
    assert not np.allclose(pdf_ttr, pdf_raw, rtol=0.1)


@pytest.mark.parametrize(
    "transformer",
    [
        MinMaxScaler(),
        FunctionTransformer(func=expit, inverse_func=logit),
    ],
)
def test_transformer_works_on_different_df_than_fit(sample_data, transformer):
    """Test transformer works when transform is called on different data than fit."""
    np.random.seed(123)
    _, y_train = sample_data
    size = 500
    y_test = pd.DataFrame(
        2 * np.random.normal(2, 1.5, (size, 1))
        + np.random.normal(0, 0.5, size).reshape(-1, 1),
        columns=["target"],
        index=pd.RangeIndex(start=1000, stop=1000 + size),
    )

    diff_transformer = DifferentiableTransformer(transformer=transformer)
    diff_transformer.fit(y_train)

    y_test_transformed = diff_transformer.transform(y_test)
    assert y_test_transformed.shape == y_test.shape
    pd.testing.assert_index_equal(y_test_transformed.index, y_test.index)
    pd.testing.assert_index_equal(y_test_transformed.columns, y_test.columns)

    y_test_reconstructed = diff_transformer.inverse_transform(y_test_transformed)
    assert np.allclose(y_test.values, y_test_reconstructed.values, rtol=1e-5)
    pd.testing.assert_index_equal(y_test_reconstructed.index, y_test.index)
    pd.testing.assert_index_equal(y_test_reconstructed.columns, y_test.columns)


def test_transformer_refit_on_different_features(sample_data):
    """Test transformer can be refit on data with different features."""
    np.random.seed(789)
    _, y_train = sample_data

    mms_diff = DifferentiableTransformer(transformer=MinMaxScaler())
    mms_diff.fit(y_train)
    original_columns = mms_diff.columns_

    y_new = pd.DataFrame(np.random.normal(5, 2, (800, 1)), columns=["new_target"])
    mms_diff.fit(y_new)

    assert not mms_diff.columns_ == original_columns
    assert mms_diff.columns_ == y_new.columns

    y_multi = pd.DataFrame(
        np.random.normal(3, 1.5, (600, 3)), columns=["feat_a", "feat_b", "feat_c"]
    )

    mms_diff.fit(y_multi)
    pd.testing.assert_index_equal(mms_diff.columns_, y_multi.columns)
    pd.testing.assert_index_equal(mms_diff.transform(y_multi).columns, y_multi.columns)
