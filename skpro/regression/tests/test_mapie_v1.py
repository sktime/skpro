import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


@pytest.mark.skipif(
    not _check_soft_dependencies("mapie>=1.0", severity="none"),
    reason="mapie>=1.0 not installed",
)
def test_mapie_v1_imports():
    """Test imports from the new conformal/jackknife modules."""
    from skpro.regression.conformal import (
        MapieConformalizedQuantileRegressor,
        MapieCrossConformalRegressor,
        MapieSplitConformalRegressor,
    )
    from skpro.regression.jackknife import MapieJackknifeAfterBootstrapRegressor

    assert MapieSplitConformalRegressor is not None
    assert MapieCrossConformalRegressor is not None
    assert MapieJackknifeAfterBootstrapRegressor is not None
    assert MapieConformalizedQuantileRegressor is not None


@pytest.mark.skipif(
    not _check_soft_dependencies("mapie>=1.0", severity="none"),
    reason="mapie>=1.0 not installed",
)
def test_mapie_v1_imports_from_top_level():
    """Test imports from top-level regression module."""
    from skpro.regression import (
        MapieConformalizedQuantileRegressor,
        MapieCrossConformalRegressor,
        MapieJackknifeAfterBootstrapRegressor,
        MapieSplitConformalRegressor,
    )

    assert MapieSplitConformalRegressor is not None
    assert MapieCrossConformalRegressor is not None
    assert MapieJackknifeAfterBootstrapRegressor is not None
    assert MapieConformalizedQuantileRegressor is not None


@pytest.mark.skipif(
    not _check_soft_dependencies("mapie>=1.0", severity="none"),
    reason="mapie>=1.0 not installed",
)
@pytest.mark.parametrize(
    "estimator_class",
    [
        "MapieSplitConformalRegressor",
        "MapieCrossConformalRegressor",
        "MapieJackknifeAfterBootstrapRegressor",
        "MapieConformalizedQuantileRegressor",
    ],
)
def test_mapie_v1_fit_predict(estimator_class):
    import skpro.regression as regression

    # Get class from module
    cls = getattr(regression, estimator_class)

    # Create dummy data - CQR needs more samples for calibration
    n_samples = 200 if estimator_class == "MapieConformalizedQuantileRegressor" else 100
    X, y = make_regression(
        n_samples=n_samples, n_features=5, noise=0.1, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(5)])
    y = pd.DataFrame(y, columns=["target"])

    # Instantiate
    if estimator_class == "MapieConformalizedQuantileRegressor":
        # CQR needs a quantile regressor and uses confidence_level at training
        # Also needs more calibration samples
        from sklearn.ensemble import GradientBoostingRegressor

        est = cls(
            estimator=GradientBoostingRegressor(loss="quantile", alpha=0.5),
            confidence_level=0.9,  # Train for 90% confidence
            test_size=0.3,  # More calibration samples
            random_state=42,
        )
    else:
        est = cls(estimator=LinearRegression(), random_state=42)

    # Fit
    est.fit(X, y)

    # Predict
    y_pred = est.predict(X)
    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.shape == (n_samples, 1)
    assert list(y_pred.columns) == ["target"]

    # Predict Interval
    coverage = [0.9, 0.95]
    y_pred_int = est.predict_interval(X, coverage=coverage)

    assert isinstance(y_pred_int, pd.DataFrame)
    assert y_pred_int.shape == (n_samples, 4)  # 2 coverages * 2 bounds

    # Check MultiIndex columns
    assert isinstance(y_pred_int.columns, pd.MultiIndex)
    # Levels: variable, coverage, bound
    assert len(y_pred_int.columns.levels) == 3
    assert set(y_pred_int.columns.get_level_values(0)) == {"target"}
    assert set(y_pred_int.columns.get_level_values(1)) == {0.9, 0.95}
    assert set(y_pred_int.columns.get_level_values(2)) == {"lower", "upper"}

    # Check values
    # Lower bound should be <= Upper bound
    for cov in coverage:
        lower = y_pred_int[("target", cov, "lower")]
        upper = y_pred_int[("target", cov, "upper")]
        assert (lower <= upper).all()
