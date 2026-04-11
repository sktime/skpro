import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skpro.distributions.normal import Normal
from skpro.regression.compose import DistrPredictiveCalibration, TargetTransform
from skpro.regression.residual import ResidualDouble


class _ShiftScaleCalibrator(BaseEstimator):
    """Calibrator that shifts location and scales spread for test assertions."""

    def __init__(self, offset=5.0, spread_mult=1.25):
        self.offset = offset
        self.spread_mult = spread_mult

    def fit(self, y_true, y_pred):
        self.shift_ = float(self.offset)
        return self

    def transform(self, y_pred):
        if hasattr(y_pred, "mu") and hasattr(y_pred, "sigma"):
            mu = np.asarray(y_pred.mu) + self.shift_
            sigma = np.asarray(y_pred.sigma) * self.spread_mult
            return Normal(mu=mu, sigma=sigma, index=y_pred.index, columns=y_pred.columns)
        if isinstance(y_pred, pd.DataFrame):
            return y_pred * self.spread_mult + self.shift_
        return y_pred


@pytest.fixture
def diabetes_split():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    return train_test_split(X, y, test_size=0.3, random_state=42)


def _qnorm_84():
    return 0.8413447460685429


def test_target_transform_standard_scaler_inverse_transforms_location_and_scale(
    diabetes_split,
):
    X_train, X_test, y_train, _ = diabetes_split

    base_reg = ResidualDouble(estimator=LinearRegression())
    scaler = StandardScaler()

    wrapped = TargetTransform(regressor=clone(base_reg), transformer=StandardScaler())
    wrapped.fit(X_train, y_train)

    y_train_t = pd.DataFrame(
        scaler.fit_transform(y_train), index=y_train.index, columns=y_train.columns
    )
    manual_reg = clone(base_reg)
    manual_reg.fit(X_train, y_train_t)

    wrapped_dist = wrapped.predict_proba(X_test)
    manual_dist_t = manual_reg.predict_proba(X_test)

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])

    expected_mu = np.asarray(manual_dist_t.mu) * scale + mean
    expected_sigma = np.asarray(manual_dist_t.sigma) * scale

    wrapped_median = wrapped_dist.ppf(0.5).to_numpy()
    wrapped_q84 = wrapped_dist.ppf(_qnorm_84()).to_numpy()
    wrapped_sigma = wrapped_q84 - wrapped_median

    assert np.allclose(wrapped_median, expected_mu, rtol=1e-6, atol=1e-6)
    assert np.allclose(wrapped_sigma, expected_sigma, rtol=1e-6, atol=1e-6)


def test_distr_predictive_calibration_modifies_predicted_distribution(diabetes_split):
    X_train, X_test, y_train, _ = diabetes_split

    base_reg = ResidualDouble(estimator=LinearRegression())
    base_reg.fit(X_train, y_train)

    calibrated = DistrPredictiveCalibration(
        regressor=ResidualDouble(estimator=LinearRegression()),
        calibrator=_ShiftScaleCalibrator(spread_mult=1.4),
    )
    calibrated.fit(X_train, y_train)

    before = base_reg.predict_proba(X_test)
    after = calibrated.predict_proba(X_test)

    before_median = before.ppf(0.5).to_numpy()
    after_median = after.ppf(0.5).to_numpy()

    before_spread = before.ppf(_qnorm_84()).to_numpy() - before_median
    after_spread = after.ppf(_qnorm_84()).to_numpy() - after_median

    assert not np.allclose(after_median, before_median)
    assert not np.allclose(after_spread, before_spread)


def test_distr_predictive_calibration_modifies_quantile_predictions(diabetes_split):
    X_train, X_test, y_train, _ = diabetes_split

    base_reg = ResidualDouble(estimator=LinearRegression())
    base_reg.fit(X_train, y_train)

    calibrated = DistrPredictiveCalibration(
        regressor=ResidualDouble(estimator=LinearRegression()),
        calibrator=_ShiftScaleCalibrator(spread_mult=1.3),
    )
    calibrated.fit(X_train, y_train)

    alpha = [0.1, 0.5, 0.9]
    before_q = base_reg.predict_quantiles(X_test, alpha=alpha)
    after_q = calibrated.predict_quantiles(X_test, alpha=alpha)

    assert not np.allclose(after_q.to_numpy(), before_q.to_numpy())
