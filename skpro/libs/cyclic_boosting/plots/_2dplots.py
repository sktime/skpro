
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from skpro.libs.cyclic_boosting.plots._1dplots import _get_y_axis

from .plot_utils import add_missing_values_box, blue_cyan_green_cmap, colorful_histogram


_imshow_style = dict(aspect="auto", origin="lower", interpolation="nearest")
_imshow_style_precisions = _imshow_style.copy()
_imshow_style_precisions["cmap"] = plt.get_cmap("gray_r")
_imshow_style_factors = _imshow_style.copy()
_imshow_style_factors["cmap"] = blue_cyan_green_cmap()


def _no_finite_samples(ax):
    plt.text(
        0.5,
        0.5,
        "No finite samples to plot",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )


def _imshow_factors_2d(ax, factors, nan_factor, nan_uncertainty, title, clim, feature, nan_count):
    """Factor plots for unsmoothed and smoothed 2d factors.
    Parameters
    ----------
    ax
        Matplotlib axis where drawing takes place
    factors
        numpy ndarray (two-dimensional) with factor data
    nan_Factor
        factor value of the NaN-bin
    nan_uncertainty
        uncertainty of the factor in the NaN-bin, optionally None
    title: str
        the title of the plot
    clim: tuple
        limits for imshow
    feature
        Feature object for the feature that is shown
    """
    first_feature_name = feature.feature_group[0] + (feature.feature_type or "")
    second_feature_name = feature.feature_group[1] + (feature.feature_type or "")

    plt.sca(ax)
    if len(factors) == 0:
        _no_finite_samples(ax)
    else:
        img = ax.imshow(factors.T, **_imshow_style_factors)
        img.set_clim(clim)
        plt.colorbar(img)

    ax.set_title(title, fontsize=9, loc="left")
    plt.xlabel(first_feature_name)
    plt.ylabel(second_feature_name)
    if np.abs(nan_factor) > 0.0001:
        add_missing_values_box(ax, nan_factor, nan_uncertainty, nan_count)


def bin_boundaries_for_factor_histograms(n_factors, extremal_absolute_factor):
    """Returns bin boundaries for a given factor array. The resulting bins are
    uneven in number, so that the neutral factor is in the center of a bin."""
    from skpro.libs.cyclic_boosting.plots import _guess_suitable_number_of_histogram_bins

    n_bins = _guess_suitable_number_of_histogram_bins(n_factors)
    if n_bins % 2 == 0:
        n_bins += 1
    return np.linspace(-extremal_absolute_factor, extremal_absolute_factor, n_bins + 1)


def _plot_factors_histogram(ax, factors, extremal_absolute_factor):
    plt.sca(ax)
    bin_boundaries = bin_boundaries_for_factor_histograms(len(factors), extremal_absolute_factor)
    freq, bin_borders = colorful_histogram(ax, factors, bin_boundaries=bin_boundaries, cmap=blue_cyan_green_cmap())
    ax.set_title("histogram of smoothed factors")
    plt.xlabel("Factor")
    ticks = 0.5 * bin_borders[1:] + 0.5 * bin_borders[:-1]
    plt.xticks(ticks, ["{:.1f}".format(x) for x in 2**ticks])


def plot_factor_2d(n_bins_finite, feature, grid_item=None):
    """
    Plots a single two dimensional factor plot. For an example see the
    :ref:`cyclic_boosting_analysis_plots`

    Parameters
    ----------
    n_bins_finite: int
        Number of finite bins
    feature: cyclic_boosting.base.Feature
        Feature as it can be obtained from the plotting observers
        ``features`` property.
    grid_item: :class:`matplotlib.gridspec.SubplotSpec`
        If the plot should be imbedded into a larger grid.
    """
    from skpro.libs.cyclic_boosting.plots import _format_groupname_with_type

    plot_yp = True
    if feature.y is None:
        plot_yp = False
    if plot_yp:
        y2d = feature.y
        prediction2d = feature.prediction
    if plot_yp:
        factors = feature.mean_dev
    else:
        factors = feature.unfitted_factors_link

    smoothed_factors = feature.factors_link
    uncertainties = feature.unfitted_uncertainties_link
    _ = _format_groupname_with_type(feature.feature_group, feature.feature_type)
    weights = feature.bin_weightsums

    nan_factor_unfitted = feature.unfitted_factor_link_nan_bin
    nan_factor = feature.factor_link_nan_bin
    nan_uncertainty = uncertainties[-1]

    def extremal_factor(x):
        return max(np.abs(np.max(x)), np.abs(np.min(x)))

    factors2d = np.reshape(factors[:-1], n_bins_finite)
    smoothed2d = np.reshape(smoothed_factors[:-1], n_bins_finite)
    if plot_yp:
        y2d = np.reshape(y2d[:-1], n_bins_finite)
        prediction2d = np.reshape(prediction2d[:-1], n_bins_finite)

    if np.prod(n_bins_finite) == 0:
        extremal_absolute_factor = 1
    else:
        extremal_absolute_factor = extremal_factor(smoothed2d)

    if grid_item is not None:
        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=grid_item)
    else:
        gs = gridspec.GridSpec(2, 2)

    clim = (-extremal_absolute_factor, extremal_absolute_factor)

    _imshow_factors_2d(
        plt.subplot(gs[0, 0]),
        factors2d,
        nan_factor=nan_factor_unfitted,
        nan_uncertainty=nan_uncertainty,
        title="final deviation of predition and truth in link space",
        clim=clim,
        feature=feature,
        nan_count=0 if len(feature.bin_weightsums) < 1 else feature.nan_bin_weightsum,
    )

    _imshow_factors_2d(
        plt.subplot(gs[1, 0]),
        smoothed2d,
        nan_factor=nan_factor,
        nan_uncertainty=None,
        title="smoothed parameters in link space",
        clim=clim,
        feature=feature,
        nan_count=0 if len(feature.bin_weightsums) < 1 else feature.nan_bin_weightsum,
    )

    def plot_marginal(axis):
        w1 = np.reshape(weights[:-1], n_bins_finite)
        w1[w1 == 0.0] = 1.0
        w = np.sum(w1, axis=axis)

        def f(x):
            return np.log(np.sum(w1 * np.exp(x), axis=axis) / w)

        marginal_smoothed = f(smoothed2d)
        marginal_dev = f(factors2d)
        if plot_yp:
            marginal_y = f(y2d)
            marginal_p = f(prediction2d)

        plt.axhline(0, color="gray")
        plt.plot(marginal_smoothed, "r.-", label="smoothed factors")
        if plot_yp:
            plt.plot(marginal_p, ".-", label="prediction", alpha=0.5)
            plt.plot(marginal_y, ".-", label="truth", alpha=0.5)
        else:
            plt.plot(marginal_dev, ".-", label="prediction / truth")
        if axis == 0:
            plt.plot(smoothed2d.T, "b-", alpha=0.03)
        else:
            plt.plot(smoothed2d, "b-", alpha=0.03)
        if plot_yp:
            all_arrays = np.r_[
                marginal_smoothed,
                marginal_dev,
                marginal_p,
                marginal_y,
                smoothed2d.flatten(),
            ]
        else:
            all_arrays = np.r_[marginal_smoothed, marginal_dev, smoothed2d.flatten()]
        minmax = np.r_[np.min(all_arrays), np.max(all_arrays)]
        y_axis_range, y_axis_labels = _get_y_axis(np.r_[minmax, marginal_smoothed])
        plt.yticks(y_axis_range, y_axis_labels)
        plt.title("Marginal Distribution: {}".format(feature.feature_group[0 if axis == 1 else 1]))
        plt.legend()

    plt.sca(plt.subplot(gs[0, 1]))
    if len(factors2d) == 0:
        _no_finite_samples(plt.gca())
    else:
        plot_marginal(1)

    plt.sca(plt.subplot(gs[1, 1]))
    if len(factors2d) == 0:
        _no_finite_samples(plt.gca())
    else:
        plot_marginal(0)


__all__ = []
