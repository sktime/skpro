"""Tests for outlier detection functionality."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest
from skbase._exceptions import NotFittedError

from skpro.outlier import (
    DensityOutlierDetector,
    LossOutlierDetector,
    QuantileOutlierDetector,
)
from skpro.outlier.base import BaseOutlierDetector
from skpro.regression.residual import ResidualDouble
from skpro.tests.test_switch import run_test_for_class


@pytest.fixture
def simple_regression_data():
    """Create simple regression data with outliers."""
    np.random.seed(42)
    n_samples = 100

    # Generate normal data
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5

    # Add some outliers
    outlier_indices = [10, 25, 50, 75, 90]
    y[outlier_indices] += np.random.randn(len(outlier_indices)) * 10

    X_df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    y_series = pd.Series(y, name="y")

    return X_df, y_series, outlier_indices


class _FallbackCRPSDistribution:
    """Minimal distribution stub for exercising fallback CRPS logic."""

    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def mean(self):
        return self._mean

    def std(self):
        return self._std


class _DummyPredictProbaRegressor:
    """Minimal regressor stub for testing score normalization."""

    _is_fitted = True

    def predict_proba(self, X):
        return object()


class _YRecordingOutlierDetector(BaseOutlierDetector):
    """Detector stub that records the internal y representation."""

    def _compute_decision_scores(self, X, y=None):
        self.last_X_type_ = type(X)
        self.last_y_type_ = type(y)
        self.last_y_ = y
        return np.zeros(len(X))


def test_base_outlier_detector_requires_fit_for_inference():
    """decision_function and predict should raise a not-fitted error before fit."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y = pd.DataFrame({"y": [1.0, 2.0, 3.0]})

    detector = _YRecordingOutlierDetector(
        regressor=_DummyPredictProbaRegressor(), contamination=0.1
    )

    with pytest.raises(NotFittedError, match="has not been fitted yet"):
        detector.decision_function(X, y)

    with pytest.raises(NotFittedError, match="has not been fitted yet"):
        detector.predict(X, y)


@pytest.mark.skipif(
    not run_test_for_class(QuantileOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_quantile_outlier_detector_fit(simple_regression_data):
    """Test fitting QuantileOutlierDetector."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = QuantileOutlierDetector(regressor, contamination=0.1)
    detector.fit(X_df, y_series)

    assert hasattr(detector, "decision_scores_")
    assert hasattr(detector, "threshold_")
    assert len(detector.decision_scores_) == len(X_df)


@pytest.mark.skipif(
    not run_test_for_class(QuantileOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_quantile_outlier_detector_predict(simple_regression_data):
    """Test prediction with QuantileOutlierDetector."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = QuantileOutlierDetector(regressor, contamination=0.1)
    detector.fit(X_df, y_series)

    predictions = detector.predict(X_df, y_series)

    assert len(predictions) == len(X_df)
    assert set(predictions).issubset({0, 1})
    # Check that approximately contamination proportion are outliers
    assert 5 <= np.sum(predictions) <= 15  # 10% +/- 5%


@pytest.mark.skipif(
    not run_test_for_class(QuantileOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_quantile_outlier_detector_decision_function(simple_regression_data):
    """Test decision function of QuantileOutlierDetector."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = QuantileOutlierDetector(regressor, contamination=0.1)
    detector.fit(X_df, y_series)

    scores = detector.decision_function(X_df, y_series)

    assert len(scores) == len(X_df)
    assert np.all(scores >= 0)  # Scores should be non-negative


@pytest.mark.skipif(
    not run_test_for_class(DensityOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_density_outlier_detector_fit(simple_regression_data):
    """Test fitting DensityOutlierDetector."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = DensityOutlierDetector(regressor, contamination=0.1)
    detector.fit(X_df, y_series)

    assert hasattr(detector, "decision_scores_")
    assert hasattr(detector, "threshold_")
    assert len(detector.decision_scores_) == len(X_df)


@pytest.mark.skipif(
    not run_test_for_class(DensityOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_density_outlier_detector_predict(simple_regression_data):
    """Test prediction with DensityOutlierDetector."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = DensityOutlierDetector(regressor, contamination=0.1)
    detector.fit(X_df, y_series)

    predictions = detector.predict(X_df, y_series)

    assert len(predictions) == len(X_df)
    assert set(predictions).issubset({0, 1})
    # Check that approximately contamination proportion are outliers
    assert 5 <= np.sum(predictions) <= 15  # 10% +/- 5%


@pytest.mark.skipif(
    not run_test_for_class(DensityOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_density_outlier_detector_use_log_false(simple_regression_data):
    """Test DensityOutlierDetector with use_log=False."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = DensityOutlierDetector(regressor, contamination=0.1, use_log=False)
    detector.fit(X_df, y_series)

    predictions = detector.predict(X_df, y_series)

    assert len(predictions) == len(X_df)
    assert set(predictions).issubset({0, 1})


@pytest.mark.skipif(
    not run_test_for_class(LossOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_loss_outlier_detector_fit(simple_regression_data):
    """Test fitting LossOutlierDetector."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = LossOutlierDetector(regressor, contamination=0.1, loss="log_loss")
    detector.fit(X_df, y_series)

    assert hasattr(detector, "decision_scores_")
    assert hasattr(detector, "threshold_")
    assert len(detector.decision_scores_) == len(X_df)


@pytest.mark.skipif(
    not run_test_for_class(LossOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_loss_outlier_detector_predict(simple_regression_data):
    """Test prediction with LossOutlierDetector."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = LossOutlierDetector(regressor, contamination=0.1, loss="log_loss")
    detector.fit(X_df, y_series)

    predictions = detector.predict(X_df, y_series)

    assert len(predictions) == len(X_df)
    assert set(predictions).issubset({0, 1})
    # Check that approximately contamination proportion are outliers
    assert 5 <= np.sum(predictions) <= 15  # 10% +/- 5%


@pytest.mark.skipif(
    not run_test_for_class(LossOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_loss_outlier_detector_crps(simple_regression_data):
    """Test LossOutlierDetector with CRPS loss."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = LossOutlierDetector(regressor, contamination=0.1, loss="crps")
    detector.fit(X_df, y_series)

    predictions = detector.predict(X_df, y_series)

    assert len(predictions) == len(X_df)
    assert set(predictions).issubset({0, 1})


def test_loss_outlier_detector_crps_fallback_multioutput_shape():
    """Fallback CRPS must reduce over outputs and return one score per sample."""
    y_true = pd.DataFrame([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], columns=["y0", "y1"])
    y_pred_dist = _FallbackCRPSDistribution(
        mean=pd.DataFrame([[0.1, 0.9], [0.8, 2.1], [2.2, 2.7]], columns=["y0", "y1"]),
        std=pd.DataFrame([[1.0, 0.5], [0.9, 0.7], [1.1, 0.6]], columns=["y0", "y1"]),
    )

    detector = LossOutlierDetector(regressor=None, loss="crps")
    scores = detector._compute_crps(y_true, y_pred_dist)

    assert scores.shape == (len(y_true),)
    assert np.isfinite(scores).all()


def test_loss_outlier_detector_crps_fallback_singleoutput_shape():
    """Fallback CRPS must preserve sample count for a single output column."""
    y_true = pd.DataFrame([[0.0], [1.0], [2.0]], columns=["y"])
    y_pred_dist = _FallbackCRPSDistribution(
        mean=pd.DataFrame([[0.1], [0.8], [2.2]], columns=["y"]),
        std=pd.DataFrame([[1.0], [0.9], [1.1]], columns=["y"]),
    )

    detector = LossOutlierDetector(regressor=None, loss="crps")
    scores = detector._compute_crps(y_true, y_pred_dist)

    assert scores.shape == (len(y_true),)
    assert np.isfinite(scores).all()


@pytest.mark.skipif(
    not run_test_for_class(LossOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_loss_outlier_detector_interval_score(simple_regression_data):
    """Test LossOutlierDetector with interval score loss."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = LossOutlierDetector(regressor, contamination=0.1, loss="interval_score")
    detector.fit(X_df, y_series)

    predictions = detector.predict(X_df, y_series)

    assert len(predictions) == len(X_df)
    assert set(predictions).issubset({0, 1})


@pytest.mark.skipif(
    not run_test_for_class(LossOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_loss_outlier_detector_custom_loss(simple_regression_data):
    """Test LossOutlierDetector with custom loss function."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    # Custom loss: absolute error
    def custom_loss(y_true, y_pred_dist):
        y_pred = y_pred_dist.mean()
        if isinstance(y_pred, (pd.DataFrame, pd.Series)):
            y_pred = y_pred.values
        if isinstance(y_true, (pd.DataFrame, pd.Series)):
            y_true = y_true.values
        return np.abs(y_true.flatten() - y_pred.flatten())

    detector = LossOutlierDetector(regressor, contamination=0.1, loss=custom_loss)
    detector.fit(X_df, y_series)

    predictions = detector.predict(X_df, y_series)

    assert len(predictions) == len(X_df)
    assert set(predictions).issubset({0, 1})


def test_loss_outlier_detector_custom_loss_multioutput_reduced_per_sample():
    """Callable loss returning (n_samples, n_outputs) should reduce to 1D."""
    X_df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y_df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})

    def custom_loss(y_true, y_pred_dist):
        del y_pred_dist
        return np.column_stack([y_true.iloc[:, 0].to_numpy(), np.ones(len(y_true))])

    detector = LossOutlierDetector(
        _DummyPredictProbaRegressor(), contamination=0.1, loss=custom_loss
    )
    scores = detector._compute_decision_scores(X_df, y_df)

    assert scores.shape == (len(X_df),)
    expected = np.sum(
        np.column_stack([y_df.iloc[:, 0].to_numpy(), np.ones(len(y_df))]), axis=1
    )
    np.testing.assert_allclose(scores, expected)


def test_loss_outlier_detector_custom_loss_multioutput_mean_aggregation():
    """Callable loss should support mean aggregation across outputs."""
    X_df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y_df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})

    def custom_loss(y_true, y_pred_dist):
        del y_pred_dist
        return np.column_stack([y_true.iloc[:, 0].to_numpy(), np.ones(len(y_true))])

    detector = LossOutlierDetector(
        _DummyPredictProbaRegressor(),
        contamination=0.1,
        loss=custom_loss,
        output_agg="mean",
    )
    scores = detector._compute_decision_scores(X_df, y_df)

    assert scores.shape == (len(X_df),)
    expected = np.mean(
        np.column_stack([y_df.iloc[:, 0].to_numpy(), np.ones(len(y_df))]), axis=1
    )
    np.testing.assert_allclose(scores, expected)


def test_loss_outlier_detector_custom_loss_invalid_shape_raises():
    """Callable loss must raise clearly when it does not return per-sample scores."""
    X_df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y_df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})

    def custom_loss(y_true, y_pred_dist):
        del y_true, y_pred_dist
        return np.array([1.0, 2.0])

    detector = LossOutlierDetector(
        _DummyPredictProbaRegressor(), contamination=0.1, loss=custom_loss
    )

    with pytest.raises(ValueError, match="one score per sample"):
        detector._compute_decision_scores(X_df, y_df)


def test_base_outlier_detector_normalizes_y_to_dataframe():
    """fit and decision_function should use DataFrame y internally."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 2.0, 3.0])

    detector = _YRecordingOutlierDetector(
        regressor=_DummyPredictProbaRegressor(), contamination=0.1
    )

    detector.fit(X, y)
    assert detector.last_X_type_ is pd.DataFrame
    assert detector.last_y_type_ is pd.DataFrame
    assert detector.last_y_.shape == (len(y), 1)

    scores = detector.decision_function(X, y)
    assert detector.last_X_type_ is pd.DataFrame
    assert detector.last_y_type_ is pd.DataFrame
    assert detector.last_y_.shape == (len(y), 1)
    assert scores.shape == (len(y),)


@pytest.mark.skipif(
    not run_test_for_class(DensityOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_outlier_fit_clones_regressor_no_side_effect(simple_regression_data):
    """fit should clone regressor and avoid mutating user-passed estimator."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    assert not regressor._is_fitted

    detector = DensityOutlierDetector(regressor, contamination=0.1)
    detector.fit(X_df, y_series)

    assert detector.regressor_ is not regressor
    assert detector.regressor_._is_fitted
    assert not regressor._is_fitted


@pytest.mark.skipif(
    not run_test_for_class(QuantileOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_quantile_outlier_detector_no_y_error(simple_regression_data):
    """Test that QuantileOutlierDetector raises error without y."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())
    regressor.fit(X_df, y_series)

    detector = QuantileOutlierDetector(regressor, contamination=0.1)

    with pytest.raises(ValueError, match="Target variable y is required"):
        detector.fit(X_df)


@pytest.mark.skipif(
    not run_test_for_class(DensityOutlierDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_density_outlier_detector_no_y_error(simple_regression_data):
    """Test that DensityOutlierDetector raises error without y in decision_function."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = DensityOutlierDetector(regressor, contamination=0.1)
    detector.fit(X_df, y_series)

    with pytest.raises(ValueError, match="Target variable y is required"):
        detector.decision_function(X_df)


@pytest.mark.skipif(
    not all(
        run_test_for_class(cls)
        for cls in (
            QuantileOutlierDetector,
            DensityOutlierDetector,
            LossOutlierDetector,
        )
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_all_detectors_compatibility(simple_regression_data):
    """Test that all detectors work with the same interface."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data

    detectors = [
        QuantileOutlierDetector(ResidualDouble(LinearRegression()), contamination=0.1),
        DensityOutlierDetector(ResidualDouble(LinearRegression()), contamination=0.1),
        LossOutlierDetector(ResidualDouble(LinearRegression()), contamination=0.1),
    ]

    for detector in detectors:
        # Test fit
        detector.fit(X_df, y_series)

        # Test decision_function
        scores = detector.decision_function(X_df, y_series)
        assert len(scores) == len(X_df)

        # Test predict
        predictions = detector.predict(X_df, y_series)
        assert len(predictions) == len(X_df)
        assert set(predictions).issubset({0, 1})
