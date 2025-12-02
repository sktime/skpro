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
    X_arr = np.random.normal(1, 1, size).reshape(-1, 1)
    y_arr = (2 * X_arr.flatten() + np.random.normal(0, 0.5, size)).reshape(-1, 1)
    X = pd.DataFrame(X_arr, columns=["feature_0"])
    y = pd.DataFrame(y_arr, columns=["target"])
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
    pred_proba_ttr = pipe.predict_proba(X)
    pdf_ttr = pred_proba_ttr.pdf(y)

    # manual transformation
    y_transformed = mms.fit_transform(y)
    est.fit(X=X, y=y_transformed)
    pred_proba_expected = est.predict_proba(X)

    # compute PDF and adjust by Jacobian
    pdf_raw = pred_proba_expected.pdf(y_transformed)
    jacobian = np.ones_like(y) * mms.scale_
    pdf_expected = pdf_raw / jacobian

    # compare PDF values
    pdf_ttr_values = pdf_ttr.values
    pdf_expected_values = pdf_expected.values
    pdf_raw_values = pdf_raw.values

    assert np.allclose(pdf_ttr_values, pdf_expected_values, rtol=1e-2)
    assert not np.allclose(pdf_ttr_values, pdf_raw_values, rtol=1e-2)


def test_ttr_pdf_vs_nonlinear_jacobian(sample_data):
    """Compare TTR PDF with manual transformation with non-linear Jacobian."""
    X, y = sample_data
    mms = FunctionTransformer(func=expit, inverse_func=logit)

    def transform_func_diff(x):
        # Derivative of expit (sigmoid): d/dx expit(x) = expit(x) * (1 - expit(x))
        return expit(x) * (1 - expit(x))

    mms_diff = DifferentiableTransformer(
        transformer=mms, transform_func_diff=transform_func_diff
    )

    est = ARDRegression()

    # full ttr pipeline
    pipe = TransformedTargetRegressor(regressor=est, transformer=mms_diff)
    pipe.fit(X=X, y=y)
    pred_proba_ttr = pipe.predict_proba(X)
    pdf_ttr = pred_proba_ttr.pdf(y)

    # manual transformation
    y_transformed = mms.fit_transform(y)
    est.fit(X=X, y=y_transformed)
    pred_proba_expected = est.predict_proba(X)

    # compute PDF and adjust by Jacobian
    pdf_raw = pred_proba_expected.pdf(y_transformed)
    # Jacobian of forward transform (expit): d/dx expit(x) = expit(x) * (1 - expit(x))
    jacobian = transform_func_diff(y).abs()
    pdf_expected = pdf_raw / jacobian

    # nonlinear transformations deviate more so we relax the tolerance
    pdf_ttr_values = pdf_ttr.values
    pdf_expected_values = pdf_expected.values
    pdf_raw_values = pdf_raw.values

    assert np.allclose(pdf_ttr_values, pdf_expected_values, rtol=0.1)
    assert not np.allclose(pdf_ttr_values, pdf_raw_values, rtol=0.1)
