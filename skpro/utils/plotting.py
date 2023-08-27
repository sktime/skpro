# -*- coding: utf-8 -*-
"""Utility functions for plotting."""
import numpy as np
import pandas as pd

from skpro.utils.validation._dependencies import _check_soft_dependencies


def plot_crossplot_interval(y_test, y_pred, coverage=None, ax=None):
    """Probabilistic cross-plot for regression, truth vs prediction interval.

    Parameters
    ----------
    y_test : array-like, [n_samples, n_targets]
        Ground truth values
    y_pred : skpro distribution, or predict_interval return, [n_samples, n_targets]
        symmetric prediction intervals are obtained
        via the coverage parameterfrom y_pred
        Predicted values
    coverage : float, optional, default=0.9
        Coverage of the prediction interval
        Used only if y_pred a distribution
    ax : matplotlib axes, optional
        Axes to plot on, if None, a new figure is created and returned

    Returns
    -------
    ax : matplotlib axes
        Axes containing the plot
        If ax was None, a new figure is created and returned
        If ax was not None, the same ax is returned with plot added

    Example
    -------
    >>> from skpro.utils.plotting import plot_crossplot_interval
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> reg_mean = LinearRegression()
    >>> reg_resid = RandomForestRegressor()
    >>> reg_proba = ResidualDouble(reg_mean, reg_resid)
    >>>
    >>> reg_proba.fit(X, y)
    ResidualDouble(...)
    >>> y_pred = reg_proba.predict_proba(X)
    >>> plot_crossplot_interval(y, y_pred)
    """
    _check_soft_dependencies("matplotlib")

    from matplotlib import pyplot

    if hasattr(y_pred, "quantile"):
        if coverage is None:
            coverage = 0.9
        quantile_pts = [0.5 - coverage / 2, 0.5, 0.5 + coverage / 2]
        y_quantiles = y_pred.quantile(quantile_pts)
        y_mid = y_quantiles.iloc[:, 1]
        y_quantiles = y_quantiles.iloc[:, [0, 2]]
    else:
        y_quantiles = y_pred
        y_mid = y_quantiles.mean(axis=1)

    y_mid_two = pd.DataFrame([y_mid, y_mid]).values
    y_quantiles_np = y_quantiles.values.T
    y_bars = np.abs(y_mid_two - y_quantiles_np)

    if ax is None:
        _, ax = pyplot.subplots()

    ax.plot(y_test, y_test, "g.", label="Optimum")
    ax.errorbar(
        y_test.values,
        y_mid,
        yerr=y_bars,
        label="Predictions",
        fmt="b.",
        ecolor="r",
        linewidth=0.5,
    )
    ax.set_ylabel("Predicted $y_{pred}$")
    ax.set_xlabel("Correct label $y_{true}$")
    ax.legend(loc="best")

    return ax


def plot_crossplot_std(y_test, y_pred, ax=None):
    """Probabilistic cross-plot for regression, error vs predictive standard deviation.

    Parameters
    ----------
    y_test : array-like, [n_samples, n_targets]
        Ground truth values
    y_pred : skpro distribution, or predict_var return, [n_samples, n_targets]
        Predicted values
    ax : matplotlib axes, optional
        Axes to plot on, if None, a new figure is created and returned

    Returns
    -------
    ax : matplotlib axes
        Axes containing the plot
        If ax was None, a new figure is created and returned
        If ax was not None, the same ax is returned with plot added

    Example
    -------
    >>> from skpro.utils.plotting import plot_crossplot_std
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> reg_mean = LinearRegression()
    >>> reg_resid = RandomForestRegressor()
    >>> reg_proba = ResidualDouble(reg_mean, reg_resid)
    >>>
    >>> reg_proba.fit(X, y)
    ResidualDouble(...)
    >>> y_pred = reg_proba.predict_proba(X)
    >>> plot_crossplot_std(y, y_pred)
    """
    _check_soft_dependencies("matplotlib")

    from matplotlib import pyplot

    if hasattr(y_pred, "_tags"):
        y_var = y_pred.var()

    y_std = np.sqrt(y_var)

    if ax is None:
        _, ax = pyplot.subplots()

    ax.plot(
        np.abs(y_pred.mean().values.flatten() - y_test.values.flatten()),
        y_std.values.flatten(),
        "b.",
    )
    ax.set_ylabel("Predictive variance of $y_{pred}$")
    ax.set_xlabel("Absolute error $|y_{true} - y_{pred}|$")
    # ax.legend(loc="best")

    return ax
