"""Utility functions for plotting."""

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

__all__ = [
    "plot_crossplot_interval",
    "plot_crossplot_std",
    "plot_crossplot_loss",
    "plot_calibration",
]
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

    Examples
    --------
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

    Examples
    --------
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

    Examples
    --------
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


def plot_calibration(y_true, y_pred, ax=None):
    r"""Plot the calibration curve for a sample of quantile predictions.

    Visualizes calibration of the quantile predictions.

    Computes the following calibration plot:

    Let :math:`p_1, \dots, p_k` be the quantile points at which
    predictions in ``y_pred`` were queried,
    e.g., via ``alpha`` in ``predict_quantiles``.

    Let :math:`y_1, \dots, y_N` be the actual values in ``y_true``,
    and let :math:`\widehat{y}_{i,j}`, for :math:`i = 1, \dots, N, j = 1, \dots, k`
    be quantile predictions at quantile point :math:`p_j`,
    of the conditional distribution of :math:`y_i`, as contained in ``y_pred``.

    We compute the calibration indicators :math:`c_{i, j},`
    as :math:`c_{i, j} = 1, \text{ if } y_i \le \widehat{y}_{i,j} \text{ and } 0,
    \text{otherwise},` and calibration fractions as

    .. math:: \widehat{p}_j = \frac{1}{N} \sum_{i = 1}^N c_{i, j}.

    If the quantile predictions are well-calibrated, we expect :math:`\widehat{p}_j`
    to be close to :math:`p_j`.

    x-axis: interval from 0 to 1, quantile points

    y-axis: interval from 0 to 1, calibration fractions

    plot elements: calibration curve of the quantile predictions (blue) and the ideal
    calibration curve (orange), the curve with equation y = x.
        Calibration curve are points :math:`(p_i, \widehat{p}_i), i = 1 \dots, k`;

        Ideal curve is the curve with equation y = x,
        containing points :math:`(p_i, p_i)`.

    Parameters
    ----------
    y_true : pd.Series, single columned pd.DataFrame, or single columned np.array.
        The actual values
    y_pred : pd.DataFrame or BaseDistribution
        The quantile predictions, formatted as returned by ``predict_quantiles``,
        or a BaseDistribution object as returned by ``predict_proba``
    ax : matplotlib.axes.Axes, optional (default=None)
        Axes on which to plot. If None, axes will be created and returned.

    Returns
    -------
    fig : matplotlib.figure.Figure, returned only if ax is None
        matplotlib figure object
    ax : matplotlib.axes.Axes
        matplotlib axes object with the figure

    Examples
    --------
    >>> from skpro.utils.plotting import plot_calibration  # doctest: +SKIP
    >>> from skpro.regression.residual import ResidualDouble  # doctest: +SKIP
    >>> from sklearn.ensemble import RandomForestRegressor  # doctest: +SKIP
    >>> from sklearn.linear_model import LinearRegression  # doctest: +SKIP
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>> from sklearn.model_selection import train_test_split  # doctest: +SKIP
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)  # doctest: +SKIP
    >>> reg_mean = RandomForestRegressor()  # doctest: +SKIP
    >>> reg_resid = LinearRegression()  # doctest: +SKIP
    >>> reg_proba = ResidualDouble(reg_mean, reg_resid)  # doctest: +SKIP
    >>> reg_proba.fit(X_train, y_train)  # doctest: +SKIP
    ResidualDouble(...)
    >>> y_pred = reg_proba.predict_proba(X_test)  # doctest: +SKIP
    >>> plot_calibration(y_test, y_pred)  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    # handle BaseDistribution input
    if hasattr(y_pred, "quantile") and not isinstance(y_pred, pd.DataFrame):
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        y_pred = y_pred.quantile(alpha)

    # ensure y_true is a pd.Series
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true.squeeze())

    _ax_kwarg_is_none = True if ax is None else False

    if _ax_kwarg_is_none:
        fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))

    result = [0]
    ideal_calibration = [0]

    for col in y_pred.columns:
        if isinstance(col, tuple):
            q = col[1]
        else:
            q = col
        pred_q = y_pred[col].to_numpy()
        result.append(np.mean(y_true.to_numpy() < pred_q))
        ideal_calibration.append(q)

    result.append(1)
    ideal_calibration.append(1)

    df = pd.DataFrame(
        {"Forecast's Calibration": result, "Ideal Calibration": ideal_calibration},
        index=ideal_calibration,
    )

    df.plot(ax=ax)

    if _ax_kwarg_is_none:
        return fig, ax
    else:
        return ax
