"""Test functionality of time series plotting functions."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import pytest

from skpro.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency for matplotlib not available",
)
def test_plot_crossplot_interval():
    """Test that plot_crossplot_interval runs without error."""
    _check_soft_dependencies("matplotlib")

    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    from skpro.regression.residual import ResidualDouble
    from skpro.utils.plotting import plot_crossplot_interval

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    reg_mean = LinearRegression()
    reg_resid = RandomForestRegressor()
    reg_proba = ResidualDouble(reg_mean, reg_resid)

    reg_proba.fit(X, y)
    y_pred_proba = reg_proba.predict_proba(X)

    plot_crossplot_interval(y, y_pred_proba, coverage=0.8)
    plot_crossplot_interval(y, y_pred_proba)

    y_pred_interval = reg_proba.predict_interval(X, coverage=0.7)
    plot_crossplot_interval(y, y_pred_interval)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency for matplotlib not available",
)
def test_plot_crossplot_std():
    """Test that plot_crossplot_std runs without error."""
    _check_soft_dependencies("matplotlib")

    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    from skpro.regression.residual import ResidualDouble
    from skpro.utils.plotting import plot_crossplot_std

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    reg_mean = LinearRegression()
    reg_resid = RandomForestRegressor()
    reg_proba = ResidualDouble(reg_mean, reg_resid)

    reg_proba.fit(X, y)
    y_pred = reg_proba.predict_proba(X)

    plot_crossplot_std(y, y_pred)

    y_pred_var = reg_proba.predict_var(X)
    plot_crossplot_std(y, y_pred_var)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency for matplotlib not available",
)
def test_plot_crossplot_loss():
    """Test that plot_crossplot_loss runs without error."""
    _check_soft_dependencies("matplotlib")

    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    from skpro.metrics import CRPS
    from skpro.regression.residual import ResidualDouble
    from skpro.utils.plotting import plot_crossplot_loss

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    reg_mean = LinearRegression()
    reg_resid = RandomForestRegressor()
    reg_proba = ResidualDouble(reg_mean, reg_resid)

    reg_proba.fit(X, y)
    y_pred = reg_proba.predict_proba(X)

    crps_metric = CRPS()
    plot_crossplot_loss(y, y_pred, crps_metric)
