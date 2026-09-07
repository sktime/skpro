import matplotlib.pyplot as plt
import numpy as np

from skpro.libs.cyclic_boosting import flags
from skpro.libs.cyclic_boosting.link import (
    IdentityLinkMixin,
    LogitLinkMixin,
    LogLinkMixin,
)


def _format_tick(tick, precision=1e-2):
    """Returns a suitable string representation for a
    factor, when it is given a tick position in linear space.

    A "suitable" string reprentation is (in this order):

    * an integer
    * decimal numbers with precision 2
    """
    if tick - precision < np.round(tick) < tick + precision:
        return f"{tick:.0f}"
    return "{:.{prec}f}".format(tick, prec=2)


def _get_x_axis(factors, bin_bounds, is_continuous):
    """
    Get x axis range and tick labels
    """
    if bin_bounds is not None:
        if is_continuous:
            labels = np.round((bin_bounds[:-1] + bin_bounds[1:]) / 2.0, 5)
        else:
            labels = bin_bounds[1:]
    else:
        labels = None
    x_axis_range = np.arange(len(factors)).astype(np.float64)
    return x_axis_range, labels


def _get_optimal_number_of_ticks(distance):
    """
    Return optimal number of ticks given a distance of the upper and lower bound
    First we scale the distance to the interval [1,20], afterwards we search
    the smalest number out of {1,2,4,5,10,20} which leads to a total tick number in (10,20].
    If we encounter something unexpected we just return 21
    """
    if distance < 1 or np.isinf(distance) or np.isnan(distance):
        return 21
    while distance > 20:
        distance /= 10
    for n in [20, 10, 5, 4, 2, 1]:
        n_ticks = np.floor(distance * n)
        if 10 < n_ticks <= 20:
            return n_ticks + 1
    return 21


def _get_y_axis(factors, uncertainties=None):
    """
    Get y axis range and tick labels
    """
    if len(factors[:-1]) > 0:
        if uncertainties is not None:
            y_max_link = max(np.max(factors[:-1] + uncertainties[1][:-1]), factors[-1])
            y_min_link = min(np.min(factors[:-1] - uncertainties[0][:-1]), factors[-1])
        else:
            y_max_link = np.max(factors)
            y_min_link = np.min(factors)
    else:
        y_max_link = factors[-1] + 0.5
        y_min_link = factors[-1] - 0.5

    y_min_link_int = np.floor(y_min_link)
    y_max_link_int = np.ceil(y_max_link)

    distance_int = y_max_link_int - y_min_link_int
    n_ticks = _get_optimal_number_of_ticks(distance_int)

    linspace = np.linspace(y_min_link_int, y_max_link_int, int(n_ticks))

    return linspace, list(map(_format_tick, linspace))


def _ensure_tuple(x):
    """
    Ensures that given object is a tuple, if not wrap it in a tuple
    """
    return x if isinstance(x, tuple) else (x,)


def _plot_factors(factors, x_axis_range, label, uncertainties=None):
    """
    Plot unsmoothed factors in given range with errobars if uncertainties are provided
    """
    if uncertainties is not None:
        unsmoothed_style = dict(
            capsize=2.5, markersize=2, fmt="o", color="k", alpha=0.6
        )
        unsmoothed_style["label"] = label
        plt.errorbar(x_axis_range, factors, yerr=uncertainties, **unsmoothed_style)
    else:
        unsmoothed_style = dict(markersize=2, marker="o", color="k", alpha=0.6)
        unsmoothed_style["label"] = label
        plt.plot(x_axis_range, factors, **unsmoothed_style)


def _plot_smoothed_factors(factors, x_axis_range, is_continuous, uncertainties=None):
    """
    Plot smoothed factors, plot style depends on is_continuous
    """
    if is_continuous:
        smoothed_style = dict(linestyle="-", linewidth=1.0, color="r")
        x_axis_range = (
            x_axis_range + np.append(x_axis_range, x_axis_range[-1] + 1)[1:]
        ) / 2
    else:
        smoothed_style = dict(
            marker="o",
            markeredgecolor="r",
            markersize=5.0,
            linestyle="none",
            fillstyle="none",
            color="r",
        )
    smoothed_style["label"] = "smoothed factors"
    if uncertainties is not None:
        plt.errorbar(
            x_axis_range,
            factors,
            yerr=[uncertainties[0], uncertainties[1]],
            **smoothed_style,
        )
    else:
        plt.plot(x_axis_range, factors, **smoothed_style)


def _plot_missing_factor(factors, x_axis_range, y_axis_range):
    """
    Plot the factor which was calculated for missing values and mark the region in an orange color
    """
    # Factor which corresponds to missing value is the last one
    missing_factor = factors[-1]
    x_position = x_axis_range[-1]

    # Plot single datapoint and shade the whole area around this point to mark it as "special"
    nan_style = dict(
        marker="p",
        markeredgecolor="r",
        markersize=5.0,
        color="b",
        linestyle="none",
        fillstyle="none",
    )
    nan_style["label"] = "smoothed nan factor"
    plt.plot([x_position], [missing_factor], **nan_style)
    plt.fill_between(
        [x_position - 0.5, x_position + 0.5],
        min(y_axis_range),
        max(y_axis_range),
        color="#f7d208",
        alpha=0.5,
    )


def _plot_axes(x_axis_range, x_axis_labels, y_axis_range, y_axis_labels, is_continuous):
    """
    Plot axes including limits, labels and ticks
    """
    # Set limits
    plt.xlim(min(x_axis_range) - 0.5, max(x_axis_range) + 0.5)
    plt.ylim(min(y_axis_range), max(y_axis_range))

    # Shift x_axis to the left if continuous, because in this case the label correspond to the bin boundaries
    if is_continuous:
        x_axis_range -= 0.5
    if x_axis_labels is not None:
        if len(x_axis_range) - len(x_axis_labels) == 1:
            x_axis_labels = np.append(x_axis_labels, "")
        plt.xticks(x_axis_range, x_axis_labels, size="xx-small", rotation="vertical")
    plt.yticks(y_axis_range, y_axis_labels)


def plot_factor_1d(
    feature,
    bin_bounds=None,
    with_errorbars=True,
    ylimits_include_errors=True,
    link_function=None,
    plot_yp=True,
):
    """
    Plots a single one dimensional factor plot.

    Parameters
    ----------
    bin_bounds: list
        Bin boundaries to label the bins.
    feature: cyclic_boosting.base.Feature
        Feature as it can be obtained from the plotting observers
        ``features`` property.
    link_function: cyclic_boosting.link.LinkFunction
        Link function of the plotted feature
    with_errorbars: bool
        Option to switch errorbars on/off.
    ylimits_include_errors: bool
        Option to show the errorbars in the plot completely.
    plot_yp: bool
        Show deviation between truth and prediction in last iteration.
    """
    y = feature.y
    if y is None:
        plot_yp = False
    p = feature.prediction

    if plot_yp:
        factors = feature.mean_dev
    else:
        factors = feature.unfitted_factors_link

    smoothed_factors = feature.factors_link

    if plot_yp:
        uncertainties = np.abs(feature.unfitted_factors_link)
        uncertainties = [uncertainties, uncertainties]
    else:
        uncertainties = [
            feature.unfitted_uncertainties_link,
            feature.unfitted_uncertainties_link,
        ]

    assert (
        len(factors)
        == len(smoothed_factors)
        == len(uncertainties[0])
        == len(uncertainties[1])
        > 0
    )
    number_of_factors = len(factors)

    if isinstance(link_function, IdentityLinkMixin):
        plt.axhline(0, color="gray")
        plt.ylabel("Summand")

    elif isinstance(link_function, LogLinkMixin):
        factors = link_function.unlink_func(factors)
        smoothed_factors = link_function.unlink_func(smoothed_factors)
        if plot_yp:
            y = link_function.unlink_func(y)
            p = link_function.unlink_func(p)
        plt.axhline(1, color="gray")
        plt.ylabel("Factor")

    elif isinstance(link_function, LogitLinkMixin):
        lower = np.abs(
            link_function.unlink_func(factors - uncertainties[0])
            - link_function.unlink_func(factors)
        )
        upper = np.abs(
            link_function.unlink_func(factors + uncertainties[1])
            - link_function.unlink_func(factors)
        )
        factors = link_function.unlink_func(factors)
        smoothed_factors = link_function.unlink_func(smoothed_factors)
        uncertainties = [
            np.where(lower < 0.0, 0.0, lower),
            np.where(upper > 1.0, 1.0, upper),
        ]
        if plot_yp:
            # do not unlink for nbinom width mode
            if ((link_function.unlink_func(y) >= 0).all()) and (
                (link_function.unlink_func(y) <= 1).all()
            ):
                y = link_function.unlink_func(y)
            p = link_function.unlink_func(p)
        plt.axhline(0.5, color="gray")
        plt.ylabel("Probability")

    else:
        plt.ylabel("Unkown")

    # Too many factors make the plot unreadable. Thus we resort to plotting a
    # histogram of factors in these cases.
    if number_of_factors > 400:
        from skpro.libs.cyclic_boosting.plots import plot_factor_histogram

        plot_factor_histogram(feature)
        return

    feature_property = _ensure_tuple(feature.feature_property)
    is_continuous = flags.is_continuous_set(feature_property[0]) | flags.is_linear_set(
        feature_property[0]
    )
    if plot_yp:
        minmax = np.r_[
            np.min(np.r_[factors, smoothed_factors, y, p]),
            np.max(np.r_[factors, smoothed_factors, y, p]),
        ]
    else:
        minmax = np.r_[
            np.min(np.r_[factors, smoothed_factors]),
            np.max(np.r_[factors, smoothed_factors]),
        ]
    f = factors.copy()
    if len(f) > 1:
        f[:2] = minmax
        u = uncertainties
    else:
        f = minmax
        u = np.c_[uncertainties, uncertainties]

    y_axis_range, y_axis_labels = _get_y_axis(f, u if ylimits_include_errors else None)
    x_axis_range, x_axis_labels = _get_x_axis(factors, bin_bounds, is_continuous)

    if "MISSING" in flags._convert_flags_to_string(feature.feature_property[0]):
        _plot_missing_factor(smoothed_factors, x_axis_range, y_axis_range)
    elif len(factors) > 1:
        factors = factors[:-1]
        uncertainties = [uncertainties[0][:-1], uncertainties[1][:-1]]
        smoothed_factors = smoothed_factors[:-1]
        x_axis_range = x_axis_range[:-1]
    _plot_axes(x_axis_range, x_axis_labels, y_axis_range, y_axis_labels, is_continuous)

    _plot_smoothed_factors(smoothed_factors, x_axis_range, is_continuous, None)

    if not plot_yp:
        label = "factors"
        if is_continuous:
            x_axis_range = (
                x_axis_range + np.append(x_axis_range, x_axis_range[-1] + 1)[1:]
            ) / 2
        _plot_factors(
            factors, x_axis_range, label, uncertainties if with_errorbars else None
        )

    try:
        if len(factors) > 1:
            plt.plot(p[:-1], ".-", label="prediction", alpha=0.5)
            plt.plot(y[:-1], ".-", label="truth", alpha=0.5)
        else:
            plt.plot(p, ".-", label="prediction", alpha=0.5)
            plt.plot(y, ".-", label="truth", alpha=0.5)
    except:
        pass

    from skpro.libs.cyclic_boosting.plots import _format_groupname_with_type

    feature_group = _format_groupname_with_type(
        feature.feature_group, feature.feature_type
    )
    plt.xlabel(feature_group)
    plt.legend()


__all__ = []
