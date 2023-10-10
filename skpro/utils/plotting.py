"""Utility functions for plotting."""
import numpy as np
import pandas as pd

from skpro.utils.validation._dependencies import _check_soft_dependencies

__authors__ = ["fkiraly", "frthjf"]


def plot_crossplot_interval(y_true, y_pred, coverage=None, ax=None):
    """Probabilistic cross-plot for regression, truth vs prediction interval.

    Plots:

    * x-axis: ground truth value
    * y-axis: median predictive value, with error bars being
      the prediction interval at symmetric coverage ``coverage``

    Parameters
    ----------
    y_true : array-like, [n_samples, n_targets]
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
    >>> from skpro.utils.plotting import plot_crossplot_interval  # doctest: +SKIP
    >>> from skpro.regression.residual import ResidualDouble  # doctest: +SKIP
    >>> from sklearn.ensemble import RandomForestRegressor  # doctest: +SKIP
    >>> from sklearn.linear_model import LinearRegression  # doctest: +SKIP
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> reg_mean = LinearRegression()  # doctest: +SKIP
    >>> reg_resid = RandomForestRegressor()  # doctest: +SKIP
    >>> reg_proba = ResidualDouble(reg_mean, reg_resid)  # doctest: +SKIP
    >>>
    >>> reg_proba.fit(X, y)  # doctest: +SKIP
    ResidualDouble(...)
    >>> y_pred = reg_proba.predict_proba(X)  # doctest: +SKIP
    >>> plot_crossplot_interval(y, y_pred)  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")

    from matplotlib import pyplot

    if hasattr(y_pred, "quantile") and not isinstance(y_pred, pd.DataFrame):
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

    ax.plot(y_true, y_true, "g.", label="Optimum")
    ax.errorbar(
        y_true.values,
        y_mid,
        yerr=y_bars,
        label="Predictions",
        fmt="b.",
        ecolor="r",
        linewidth=0.5,
    )
    ax.set_ylabel(r"Prediction interval $\widehat{y}_i$")
    ax.set_xlabel(r"Correct label $y_i$")
    ax.legend(loc="best")

    return ax


def plot_crossplot_std(y_true, y_pred, ax=None):
    r"""Probabilistic cross-plot for regression, error vs predictive standard deviation.

    Plots:

    * x-axis: absolute error samples :math:`|y_i - \widehat{y}_i.\mu|`
    * y-axis: predictive standard deviation :math:`\widehat{y}_i.\sigma`,
      of the prediction :math:`\widehat{y}_i` corresponding to :math:`y_i`

    Parameters
    ----------
    y_true : array-like, [n_samples, n_targets]
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
    >>> from skpro.utils.plotting import plot_crossplot_std  # doctest: +SKIP
    >>> from skpro.regression.residual import ResidualDouble  # doctest: +SKIP
    >>> from sklearn.ensemble import RandomForestRegressor  # doctest: +SKIP
    >>> from sklearn.linear_model import LinearRegression  # doctest: +SKIP
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> reg_mean = LinearRegression()  # doctest: +SKIP
    >>> reg_resid = RandomForestRegressor()  # doctest: +SKIP
    >>> reg_proba = ResidualDouble(reg_mean, reg_resid)  # doctest: +SKIP
    >>>
    >>> reg_proba.fit(X, y)  # doctest: +SKIP
    ResidualDouble(...)
    >>> y_pred = reg_proba.predict_proba(X)  # doctest: +SKIP
    >>> plot_crossplot_std(y, y_pred)  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")

    from matplotlib import pyplot

    if hasattr(y_pred, "_tags") and not isinstance(y_pred, pd.DataFrame):
        y_var = y_pred.var()
    else:
        y_var = y_pred

    y_std = np.sqrt(y_var)

    if ax is None:
        _, ax = pyplot.subplots()

    ax.plot(
        np.abs(y_pred.mean().values.flatten() - y_true.values.flatten()),
        y_std.values.flatten(),
        "b.",
    )
    ax.set_ylabel(r"Predictive standard deviation of $\widehat{y}_i$")
    ax.set_xlabel(r"Absolute errors $|y_i - \widehat{y}_i|$")
    # ax.legend(loc="best")

    return ax


def plot_crossplot_loss(y_true, y_pred, metric, ax=None):
    r"""Cross-loss-plot for probabilistic regression.

    Plots:

    * x-axis: ground truth values :math:`y_i`
    * y-axis: loss of the prediction :math:`\widehat{y}_i`
      corresponding to :math:`y_i`,
      as calculated by ``metric.evaluate_by_index``

    Parameters
    ----------
    y_true : array-like, [n_samples, n_targets]
        Ground truth values
    y_pred : skpro distribution, or predict_var return, [n_samples, n_targets]
        Predicted values
    metric : skpro metric
        Metric to calculate the loss
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
    >>> from skpro.utils.plotting import plot_crossplot_loss  # doctest: +SKIP
    >>> from skpro.metrics import CRPS  # doctest: +SKIP
    >>> from skpro.regression.residual import ResidualDouble  # doctest: +SKIP
    >>> from sklearn.ensemble import RandomForestRegressor  # doctest: +SKIP
    >>> from sklearn.linear_model import LinearRegression  # doctest: +SKIP
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> reg_mean = LinearRegression()  # doctest: +SKIP
    >>> reg_resid = RandomForestRegressor()  # doctest: +SKIP
    >>> reg_proba = ResidualDouble(reg_mean, reg_resid)  # doctest: +SKIP
    >>>
    >>> reg_proba.fit(X, y)  # doctest: +SKIP
    ResidualDouble(...)
    >>> y_pred = reg_proba.predict_proba(X)  # doctest: +SKIP
    >>> crps_metric = CRPS()  # doctest: +SKIP
    >>> plot_crossplot_loss(y, y_pred, crps_metric)  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")

    from matplotlib import pyplot

    losses = metric.evaluate_by_index(y_true, y_pred)
    loss_vals = losses.values.flatten()
    total_loss = np.mean(loss_vals).round(2)
    total_loss_std = np.std(loss_vals) / np.sqrt(len(loss_vals))
    total_loss_std = total_loss_std.round(2)

    overall = f"{total_loss} +/- {total_loss_std} sterr of mean"

    if ax is None:
        _, ax = pyplot.subplots()

    ax.plot(y_true, losses, "y_")

    ax.set_title(f"mean {metric.name}: {overall}")

    ax.set_xlabel(r"Correct label $y_i$")
    ax.set_ylabel(metric.name + r"($y_i$, $\widehat{y}_i$)")

    ax.tick_params(colors="y")

    return ax
