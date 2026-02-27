"""Tests for outlier detection functionality."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from skpro.outlier import (
    DensityOutlierDetector,
    LossOutlierDetector,
    QuantileOutlierDetector,
)
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


@run_test_for_class(QuantileOutlierDetector)
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


@run_test_for_class(QuantileOutlierDetector)
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


@run_test_for_class(QuantileOutlierDetector)
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


@run_test_for_class(DensityOutlierDetector)
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


@run_test_for_class(DensityOutlierDetector)
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


@run_test_for_class(DensityOutlierDetector)
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


@run_test_for_class(LossOutlierDetector)
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


@run_test_for_class(LossOutlierDetector)
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


@run_test_for_class(LossOutlierDetector)
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


@run_test_for_class(LossOutlierDetector)
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


@run_test_for_class(LossOutlierDetector)
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


@run_test_for_class(QuantileOutlierDetector)
def test_quantile_outlier_detector_no_y_error(simple_regression_data):
    """Test that QuantileOutlierDetector raises error without y."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())
    regressor.fit(X_df, y_series)

    detector = QuantileOutlierDetector(regressor, contamination=0.1)

    with pytest.raises(ValueError, match="Target variable y is required"):
        detector.fit(X_df)


@run_test_for_class(DensityOutlierDetector)
def test_density_outlier_detector_no_y_error(simple_regression_data):
    """Test that DensityOutlierDetector raises error without y in decision_function."""
    from sklearn.linear_model import LinearRegression

    X_df, y_series, _ = simple_regression_data
    regressor = ResidualDouble(LinearRegression())

    detector = DensityOutlierDetector(regressor, contamination=0.1)
    detector.fit(X_df, y_series)

    with pytest.raises(ValueError, match="Target variable y is required"):
        detector.decision_function(X_df)


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
