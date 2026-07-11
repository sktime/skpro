"""
Plots for the Cyclic Boosting family
"""

import contextlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

from skpro.libs.cyclic_boosting import CBNBinomC
from skpro.libs.cyclic_boosting.features import create_feature_id
from skpro.libs.cyclic_boosting.utils import get_bin_bounds

from ._1dplots import plot_factor_1d
from ._2dplots import plot_factor_2d
from .plot_utils import append_extension, nbpy_style


def _guess_suitable_number_of_histogram_bins(n):
    """
    Guesses a suitable number of histograms for a given number of samples.
    """
    sturges = int(np.ceil(np.log(n) / np.log(2) + 1))
    return np.clip(sturges, 5, 30)


def _factor_plots_y_limits(
    factor_arr, uncert_arr=None, percentage=0.05, ymin_for_zero=1e-3
):
    """
    >>> y = np.array([0, 1e-5, 0.1, 0.5, 1, 2, 10, 100, 1e5])
    >>> np.allclose(_factor_plots_y_limits(y), (-1e-3, 1e5 * 1.05))
    True

    >>> y = np.array([1e-5, 10])
    >>> np.allclose(_factor_plots_y_limits(y), (1e-5 * 0.95, 10 * 1.05))
    True
    """
    if uncert_arr is not None:
        ymax = np.max(factor_arr + uncert_arr)
        ymin = np.min(factor_arr - uncert_arr)
    else:
        ymax = np.max(factor_arr)
        ymin = np.min(factor_arr)

    if ymin == 0:
        ymin = -ymin_for_zero
    else:
        ymin = ymin * (1 - percentage)

    return ymin, ymax * (1 + percentage)


@nbpy_style
def plot_iteration_info(plot_observer):
    """
    Convenience method calling :func:`plot_loss` and
    :func:`plot_factor_change`.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.
    """
    plt.subplot(211)
    plot_loss(plot_observer)
    plt.subplot(212)
    plot_factor_change(plot_observer)


@nbpy_style
def plot_factor_change(plot_observer):
    """
    Plot the global factor changes for all iterations.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.
    """
    factor_changes = plot_observer.factor_change
    n = len(factor_changes)
    iterations = np.arange(1, n + 1)
    plt.xlabel("Iterations")
    plt.ylabel("Factor change of all features")
    ymin, ymax = _factor_plots_y_limits(factor_changes)
    plt.ylim(ymin, ymax)
    plt.xlim(0.9, n + 0.1)
    plt.plot(iterations, factor_changes, "ob-")
    plt.grid(True)


@nbpy_style
def plot_loss(plot_observer):
    """
    Plot the change of the loss function between the in-sample y
    and the predicted factors for all iterations.
    For the zeroth iteration the mean of y is used as prediction.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.
    """
    loss = plot_observer.loss
    n = len(loss)
    iterations = np.arange(n)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    ymin, ymax = _factor_plots_y_limits(loss)
    plt.ylim(ymin, ymax)
    plt.xlim(-0.1, n + 0.1)
    plt.plot(iterations, loss, "o-r")
    plt.grid(True)


@nbpy_style
def plot_in_sample_diagonal_plot(plot_observer):
    """
    Plot the in sample diagonal plot for prediction and truth.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.
    """
    plot_observer.check_fitted()
    means, bin_centers, errors, _ = plot_observer.histograms

    plt.plot(bin_centers, means, "o", color="b")
    ymin, ymax = plt.ylim()
    if errors is not None:
        plt.errorbar(
            bin_centers,
            means,
            yerr=[-errors[0], errors[1]],
            fmt="none",
            ecolor="b",
            capsize=2.5,
        )
    else:
        plt.errorbar(
            bin_centers,
            means,
            fmt="none",
            ecolor="b",
            capsize=2.5,
        )
    xmin, xmax = plt.xlim()
    plt.ylim(min(xmin, ymin), max(xmax, ymax))
    plt.xlim(xmin, xmax)
    plt.plot([xmin, xmax], [xmin, xmax], "k", linewidth=0.4)
    plt.xlabel("Prediction")
    plt.ylabel("Truth")


def plot_analysis(
    plot_observer,
    file_obj,
    binners=None,
    figsize=(11.69, 8.27),
    use_tightlayout=True,
    plot_yp=True,
):
    """
    Plot factors as a multipage PDF, also include plots for Loss and
    factor change behaviour and an insample diagonal plot.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.
    file: string or file-like
        If a string indicates the name of the file, to which the plots are
        written. The ending '.pdf' is added if it is not present already.
        You may also pass a file-like object.
    binners: list
        A list of binners. If binners are given the labels of the x-axis of
        factor plots are better interpretable.
    figsize: tuple
        A tuple with length containing the width and height of the figures.
    use_tightlayout: bool
        If true the tightlayout option of matplotlib is used.
    """
    filepath_or_object = append_extension(file_obj, ".pdf")
    dpi = 200
    with contextlib.closing(PdfPages(filepath_or_object)) as pdf_pages:
        plot_observer.check_fitted()

        # do not show for nbinom width mode
        if plot_observer.link_function.__class__ != CBNBinomC:
            plt.figure(figsize=figsize)
            plot_in_sample_diagonal_plot(plot_observer)
            plt.savefig(pdf_pages, format="pdf", dpi=dpi)

        for feature in plot_observer.features:
            plt.figure(figsize=figsize)
            grid = gridspec.GridSpec(1, 1)
            _plot_one_feature_group(
                plot_observer,
                grid[0],
                feature,
                binners,
                use_tightlayout,
                plot_yp=plot_yp,
            )
            plt.savefig(pdf_pages, format="pdf", dpi=dpi)

        # plt.figure(figsize=figsize)
        # plot_factor_change(plot_observer)
        # plt.savefig(pdf_pages, format="pdf", dpi=dpi)
        plt.figure(figsize=figsize)
        plot_loss(plot_observer)
        plt.savefig(pdf_pages, format="pdf", dpi=dpi)
        # for feature in plot_observer.features:
        #     plt.figure(figsize=figsize)
        #     f_sum = feature.factor_sum
        #     # f_sum /= f_sum[-1]
        #     plt.plot(f_sum)
        #     plt.title(
        #         "absolute factor sum for {} over iterations".format(
        #             feature.feature_group
        #         )
        #     )
        #     plt.savefig(pdf_pages, format="pdf", dpi=dpi)
    plt.close("all")


@nbpy_style
def plot_factors(
    plot_observer,
    binners=None,
    feature_groups_or_ids=None,
    features_per_row=2,
    use_tightlayout=True,
    plot_yp=True,
):
    """
    Create a matplotlib figure containing several factor plots (of a
    possibly pre-defined subset of features) in a grid.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.
    binners: list
        A list of binners. If binners are given the labels of the x-axis of
        factor plots are better interpretable.
    feature_groups_or_ids: list
        A list of feature group names or
        :class:`cyclic_boosting.base.FeatureID`
        for which the factors will be plotted.
        Default is to plot the factors of all features.
    features_per_row: int
        The number of factor plots in one row.
    use_tightlayout: bool
        If true the tightlayout option of matplotlib is used.
    """
    plot_observer.check_fitted()
    if feature_groups_or_ids is None:
        features = plot_observer.features
    else:
        feature_ids = [
            create_feature_id(feature_group_or_id)
            for feature_group_or_id in feature_groups_or_ids
        ]
        features = [plot_observer.features[feature_id] for feature_id in feature_ids]

    n_plots = len(features)
    grid = gridspec.GridSpec(int(np.ceil(n_plots / features_per_row)), features_per_row)

    for i, feature in enumerate(features):
        _plot_one_feature_group(
            plot_observer, grid[i], feature, binners, use_tightlayout, plot_yp=plot_yp
        )


def _format_groupname_with_type(feature_group, feature_type):
    name = ", ".join(feature_group)
    if feature_type is None:
        return name
    else:
        return f"{name} ({feature_type})"


def _plot_one_feature_group(
    plot_observer, grid_item, feature, binners=None, use_tightlayout=True, plot_yp=True
):
    if len(feature.feature_group) == 1:
        # treatment of one-dimensional features
        # no bin occupancy plot for too many bins
        if len(feature.factors_link) > 400:
            plt.subplot(grid_item)
            plot_factor_1d(
                feature,
                bin_bounds=get_bin_bounds(binners, feature.feature_group[0]),
                link_function=plot_observer.link_function,
                plot_yp=plot_yp,
            )
        else:
            gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=grid_item, height_ratios=[2.5, 0.4]
            )
            factor_plot = plt.subplot(gs[0, 0])
            plot_factor_1d(
                feature,
                bin_bounds=get_bin_bounds(binners, feature.feature_group[0]),
                link_function=plot_observer.link_function,
                plot_yp=plot_yp,
            )
            plt.subplot(gs[1, 0], sharex=factor_plot)
            bin_occupancies = np.bincount(feature.lex_binned_data)
            plt.plot(range(len(bin_occupancies)), bin_occupancies)
            plt.xticks(size="xx-small", rotation="vertical")
        plt.grid(True, which="both")

    elif len(feature.feature_group) == 2:
        # treatment of two-dimensional features
        plot_factor_2d(
            n_bins_finite=plot_observer.n_feature_bins[feature.feature_group],
            feature=feature,
            grid_item=grid_item,
        )

    else:
        plt.subplot(grid_item)
        plot_factor_high_dim(feature)

    if use_tightlayout:
        plt.tight_layout()


def plot_factor_histogram(feature):
    """
    Plots a histogram of the given factors with a logarithmic y-axis.

    Parameters
    ----------
    feature: cyclic_boosting.base.Feature
        Feature object from as it can be obtained from a plotting
        observer
    """
    factors = feature.unfitted_factors_link
    smoothed_factors = feature.factors_link
    feat_group = feature.feature_group
    feature_type = feature.feature_type
    feat_group = _format_groupname_with_type(feat_group, feature_type)
    n_bins = _guess_suitable_number_of_histogram_bins(len(factors))

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=1, figure=fig)

    fig.add_subplot(gs[0, 0])
    plt.title(f"Unsmoothed-Factor Histogram for {feat_group}")
    plt.xlabel("Factor")
    plt.ylabel("Count")
    plt.hist(factors, bins=n_bins, log=True)

    fig.add_subplot(gs[1, 0])
    dev = smoothed_factors - factors
    plt.xlabel("smoothed_factors - factors")
    plt.ylabel("Count")
    plt.hist(dev, bins=100, log=True, color="red")


plot_factor_high_dim = plot_factor_histogram


__all__ = [
    "plot_iteration_info",
    "plot_factor_change",
    "plot_loss",
    "plot_in_sample_diagonal_plot",
    "plot_analysis",
    "plot_factors",
    "plot_factor_1d",
    "plot_factor_2d",
    "plot_factor_high_dim",
    "plot_factor_histogram",
]
