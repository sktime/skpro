import contextlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from skpro.libs.cyclic_boosting import utils


def add_missing_values_box(ax, nan_value=None, nan_uncert=None, nan_count=None):
    r"""Adds a Neurobayes analysis-plot-style box to plot axes indicating the
    corresponding value for NaN-data.

    The function assumes that there is free space to the right of the plot.
    It was originally designed to fit the NaN-box between a 2-D
    :func:`matplotlib.pyplot.imshow` plot and its colorbar (see example below).

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        The axes object corresponding to the plot to which the nan box should
        be added

    nan_value : float
        A value to be displayed in the NaN box

    nan_uncert : float
        If not None, uncertainty that will be printed as plusminus deviation

    Returns
    -------

    matplotlib.axes.Axes
        axes object of the NaN-box
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    middle, right = fill_missing_values_box(ax, xmax, xmin, ymin, ymax)
    ax.set_xlim(xmin, right)

    if nan_value is None:
        nan_text = r"NaN: -"
    else:
        nan_text = r"NaN: {value:.3f}"

    if nan_uncert is not None:
        nan_text += r" $\pm$ {uncert:.3f}"

    if nan_count is not None:
        nan_text += r", #: {count:.0f}"

    ax.axvline(xmax)

    ax.text(
        middle,
        sum(ax.get_ylim()) * 0.5,
        nan_text.format(value=nan_value, uncert=nan_uncert, count=nan_count),
        rotation="vertical",
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        clip_on=True,
    )


def fill_missing_values_box(ax, xmax, xmin, ymin, ymax):
    """Fill the yellow area and return x coordinates for
    the missing values box.

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        axes object into which the missing values box will be drawn

    xmax : float
        maximum x-limit of the axes

    xmin : float
        minimum x-limit of the axes

    ymin : float
        minimum y-limit of the axes

    ymax : float
        maximum y-limit of the axes

    Note
    ----

    Be careful of the unintuitive order of the parameters.

    Returns
    -------
     (float, float)
        tuple of x-coordinates for the middle and the right edge of the box
        (the left edge is already given by xmax)
    """
    middle, right = calc_extent_with_missing_values_box(xmin, xmax)
    ax.fill_between([xmax, right], ymin, ymax, color="#f7d208", alpha=0.5)
    return middle, right


def calc_extent_with_missing_values_box(xmin, xmax):
    """Calculates new extent of a plot when a missing values box is added

    Parameters
    ----------

    xmin : float
        minimum x-limit of the original plot

    xmax : float
        maximum x-limit of the original plot

    Returns
    -------

    (float, float)
        tuple of x-coordinates for the middle and the right edge of the box
        (the left edge is already given by xmax)
    """
    step = (xmax - xmin) * 0.05
    right = xmax + step
    middle = xmax + 0.5 * step

    return middle, right


def colorful_histogram(ax, x, bin_boundaries, cmap):
    """Plot a histogram on a given matplotlib axis `ax` for to-be-binned values
    `x` with bin boundaries ``bin_boundaries`` (is passed to ``np.histogram``). Each bin
    column is colored using the map provided by ``cmap``.
    """
    freq, bin_borders = np.histogram(x, bins=bin_boundaries)
    color_norm = Normalize(vmin=np.min(bin_borders), vmax=np.max(bin_borders))
    scalar_mappable = ScalarMappable(norm=color_norm, cmap=cmap)
    bin_centers = 0.5 * bin_borders[:-1] + 0.5 * bin_borders[1:]

    ax.bar(
        bin_borders[:-1],
        freq,
        width=bin_borders[1] - bin_borders[0],
        color=scalar_mappable.to_rgba(bin_centers),
    )
    return freq, bin_borders


def blue_cyan_green_cmap():
    """Colormap from blue over cyan to green

    :rtype: :class:`matplotlib.colors.LinearSegmentedColormap`
    """
    return _colormap_gen(blue_red=False)


def _colormap_gen(blue_red=True):
    """Colormap from blue over magenta to red **or** blue over cyan to green

    Parameters
    ----------
    blue_red: bool
        If ``True`` blue-magenta-red colormap, else blue-cyan-green colormap is
        build.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Color map
    """
    r1, g1, b1 = 0, 0, 1
    r2, g2, b2 = 1, 0, 0
    if blue_red:
        color1 = "red"
        color2 = "green"
        name = "blue_magenta_red_cmap"
    else:
        color1 = "green"
        color2 = "red"
        name = "blue_cyan_green_cmap"

    cdict = {
        color1: ((0, r1, r1), (0.5, r2, r2), (1, r2, r2)),
        color2: ((0, g1, g1), (1, g2, g2)),
        "blue": ((0, b1, b1), (0.5, b1, b1), (1, b2, b2)),
    }

    cmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    if blue_red:
        cmap.set_over((0.5, 0, 0))
    else:
        cmap.set_over((0, 0.5, 0))
    cmap.set_under((0, 0, 0.5))
    return cmap


def _nbpy_style():
    """Activate nbpy's Matplotlib style default settings locally."""
    rcParams = {}
    rcParams["grid.color"] = "#000000"
    rcParams["grid.alpha"] = 0.1
    rcParams["grid.linestyle"] = "-"
    rcParams["lines.markersize"] = 3.0
    rcParams["legend.numpoints"] = 1

    # font sizes
    rcParams["font.size"] = 20
    rcParams["legend.fontsize"] = "small"
    rcParams["axes.labelsize"] = "small"
    rcParams["axes.titlesize"] = "medium"
    rcParams["xtick.labelsize"] = "small"
    rcParams["ytick.labelsize"] = "small"

    rcParams["image.interpolation"] = "antialiased"

    with mpl.rc_context(rcParams):
        yield


nbpy_style = utils.generator_to_decorator(_nbpy_style)
nbpy_style_context = contextlib.contextmanager(_nbpy_style)


@contextlib.contextmanager
def _nbpy_style_figure(num=None, figsize=None):
    """Contextmanager for providing a new plotting environment using the
    nbpy plotting style.

    Some plot functions create a new figure by default. To avoid this,
    set the corresponding keyword argument, e.g. ``fignum`` in::

        plt.matshow(matrix, fignum=False)

    Note
    ----
    In earlier versions, this context closed all pre-existing figures.
    This functionality has been removed. Please call plt.close("all") yourself,
    if you need to remove figures before plotting.

    Parameters
    ----------
    num: int
        figure number, by default :obj:`None` which, the default of
        :func:`~matplotlib.pyplot.figure` default.
    figsize: tuple, float or None
        Optional size of the new figure to create.
    """
    try:
        plt.figure(num=num, figsize=figsize)
        with nbpy_style_context():
            yield
    finally:
        pass


def append_extension(file, extension):
    """Small helper method to append possibly missing extension to a
    file name.

    Only works on instances of basestring. Leaves all other objects
    unchanged

    Parameters
    ----------

    file : any object

    extension : str
        extension (including '.') to be appended to file

    Returns
    -------
    result : object of same type as `file`
    """
    if isinstance(file, str):
        if not file.endswith(extension):
            file += extension
    return file
