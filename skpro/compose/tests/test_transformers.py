"""Tests for DifferentiableTransformer."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from skpro.compose import DifferentiableTransformer
from skpro.metrics import CRPS
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


def test_ttr_matches_manual_jacobian_adjustment(sample_data):
    """Test that TTR produces similar results to manual transformation with Jacobian.

    This test verifies that TransformedTargetRegressor correctly applies the
    change-of-variables formula by comparing it to a manual implementation where:
    1. Targets are transformed
    2. Model is fit on transformed targets
    3. Metrics are adjusted by the Jacobian of the transformation
    """
    X, y = sample_data
    mms = MinMaxScaler(feature_range=(0.1, 0.9))
    mms_diff = DifferentiableTransformer(transformer=mms)
    est = ARDRegression()

    # full ttr pipeline
    pipe = TransformedTargetRegressor(regressor=est, transformer=mms_diff)
    pipe.fit(X=X, y=y)
    pred_proba_ttr = pipe.predict_proba(X)
    crps_ttr = CRPS()(y_true=y, y_pred=pred_proba_ttr)

    # manual transformation
    y_transformed = mms.fit_transform(y)
    est.fit(X=X, y=y_transformed)
    pred_proba_expected = est.predict_proba(X)

    # compute Jacobian and adjust metric
    crps_raw = CRPS()._evaluate_by_index(y_transformed, pred_proba_expected)
    jacobian = np.ones_like(y) * mms.scale_
    crps_expected = np.mean(crps_raw / jacobian.reshape(-1, 1))

    # use the log relative CRPS to compare results
    # CRPS of the expected and TTR methods should be very similar in magnitude
    y_mean = y.mean().target
    log_rel_crps_raw = np.log((crps_raw / y_mean).mean())
    log_rel_crps_expected = np.log(crps_expected / y_mean)
    log_rel_crps_ttr = np.log(crps_ttr / y_mean)

    assert np.isclose(log_rel_crps_ttr, log_rel_crps_expected, atol=1e-2)
    assert not np.isclose(log_rel_crps_ttr, log_rel_crps_raw, atol=1e-2)
    assert not np.isclose(log_rel_crps_expected, log_rel_crps_raw, atol=1e-2)


def test_ttr_matches_nonlinear_jacobian_adjustment(sample_data):
    """Test that TTR produces similar results to manual transformation with Jacobian.

    This test verifies that TransformedTargetRegressor correctly applies the
    change-of-variables formula by comparing it to a manual implementation where:
    1. Targets are transformed
    2. Model is fit on transformed targets
    3. Metrics are adjusted by the Jacobian of the transformation
    """
    X, y = sample_data
    mms = FunctionTransformer(func=expit, inverse_func=logit)

    def inverse_func_diff(x):
        return 1 / (x * (1 - x))

    mms_diff = DifferentiableTransformer(
        transformer=mms, inverse_func_diff=inverse_func_diff
    )

    est = ARDRegression()

    # full ttr pipeline
    pipe = TransformedTargetRegressor(regressor=est, transformer=mms_diff)
    pipe.fit(X=X, y=y)
    pred_proba_ttr = pipe.predict_proba(X)
    crps_ttr = CRPS()(y_true=y, y_pred=pred_proba_ttr)

    # manual transformation
    y_transformed = mms.fit_transform(y)
    est.fit(X=X, y=y_transformed)
    pred_proba_expected = est.predict_proba(X)

    # compute Jacobian and adjust metric
    crps_raw = CRPS()._evaluate_by_index(y_transformed, pred_proba_expected)
    # Jacobian of inverse transform (logit): d/dx logit(x) = 1/(x*(1-x))
    jacobian = inverse_func_diff(y).abs().values
    crps_expected = np.mean(crps_raw / jacobian)

    # use the log relative CRPS to compare results
    # CRPS of the expected and TTR methods should be very similar in magnitude
    y_mean = y.mean().target
    log_rel_crps_raw = np.log((crps_raw / y_mean).mean())
    log_rel_crps_expected = np.log(crps_expected / y_mean)
    log_rel_crps_ttr = np.log(crps_ttr / y_mean)

    # nonlinear transformations deviate more so we relax the tolerance
    assert np.isclose(log_rel_crps_ttr, log_rel_crps_expected, atol=0.5)
    assert not np.isclose(log_rel_crps_ttr, log_rel_crps_raw, atol=0.5)
    assert not np.isclose(log_rel_crps_expected, log_rel_crps_raw, atol=0.5)
