import copy
import logging
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional

import decorator
import numba as nb
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


LOWER_1_SIGMA_QUANTILE = (1.0 - 0.6827) / 2
LOWER_2_SIGMA_QUANTILE = (1.0 - 0.9545) / 2


class ConvergenceError(RuntimeError):
    """Exception type, that should be thrown if an estimator cannot converge
    and therefore its result cannot be trusted."""


def not_seen_events(x, wx, n_bins):
    """
    Return a boolean array that slices x so that only values with
    a non-finite weightsum `wx` are used.

    Parameters
    ----------
    x: :class:`pandas.DataFrame` or :class:`numpy.ndarray`
        Array can be of any dimension. The array has to consist
        out of contigous integers.

    wx: :class:`numpy.ndarray` weightsum of all bins occuring in x

    n_bins: :class:`numpy.ndarray` of shape ``(M,)``
        number of bins in each dimension

    Example
    -------
    >>> x = np.c_[np.array([0, 1, 2, 3, 4, np.nan, -1])]
    >>> wx = np.array([3, 2, 0, 1, 0])
    >>> nbins = np.array([4, 1])
    >>> from skpro.libs.cyclic_boosting.utils import not_seen_events
    >>> not_seen_events(x, wx, nbins)
    array([False, False, True, False,  True,  True,  True], dtype=bool)

    >>> x = pd.DataFrame({"a": [0, 1, 0, 1, np.nan, 0, -1],
    ...                   "b": [0, 0, 1, 1, 1, np.nan, -1]})
    >>> wx = np.array([1, 0, 1, 0, 1])
    >>> nbins = np.array([2, 2])
    >>> not_seen_events(x, wx, nbins)
    array([False, False, True, True, False, False, False], dtype=bool)
    """
    max_lex_bins = int(np.prod(n_bins))
    not_seen = wx == 0.0

    if len(not_seen) == max_lex_bins:
        # Add nan-bin if not present yet.
        not_seen = np.r_[not_seen, True]

    finite_x = slice_finite_semi_positive(x)
    if x.shape[1] > 1:
        x, n_bins = multidim_binnos_to_lexicographic_binnos(x, n_bins)
    else:
        x = np.asarray(x).reshape(-1)
    x = np.round(x)
    x = np.asarray(x, dtype=int)
    x = np.where(~finite_x | (x > max_lex_bins), max_lex_bins, x)

    res = not_seen[x].reshape(-1)
    return res


def get_X_column(X, column, array_for_1_dim=True):
    """
    Picks columns from :class:`pandas.DataFrame` or :class:`numpy.ndarray`.

    Parameters
    ----------
    X: :class:`pandas.DataFrame` or :class:`numpy.ndarray`
        Data Source from which columns are picked.
    column:
        The format depends on the type of X. For :class:`pandas.DataFrame` you
        can give a string or a list/tuple of strings naming the columns. For
        :class:`numpy.ndarray` an integer or a list/tuple of integers indexing
        the columns.
    array_for_1_dim: bool
        In default mode (set to True) the return type for a one dimensional
        access is a np.ndarray with shape (n, ). If set to False it is a
        np.ndarray with shape (1, n).
    """
    if isinstance(column, tuple):
        column = list(column)
    if not array_for_1_dim:
        if not isinstance(column, list):
            column = [column]
    else:
        if isinstance(column, list) and len(column) == 1:
            column = column[0]
    if isinstance(X, pd.DataFrame):
        return X[column].values
    else:
        return X[:, column]


def slice_finite_semi_positive(x):
    """
    Return slice of all finite and semi positive definite values of x

    Parameters
    ----------
    x: :class:`pandas.DataFrame` or :class:`numpy.ndarray`
        Array can be of any dimension.

    Example
    -------

    >>> x = np.array([1, np.nan, 3, -2])
    >>> from skpro.libs.cyclic_boosting.utils import slice_finite_semi_positive
    >>> slice_finite_semi_positive(x)
    array([ True, False,  True, False], dtype=bool)

    >>> X = pd.DataFrame({'a': [1, np.nan, 3], 'b': [-1, 2, 3]})
    >>> slice_finite_semi_positive(X)
    array([False, False,  True], dtype=bool)
    """
    if isinstance(x, pd.DataFrame):
        x = x.values

    if len(x.shape) > 1:
        finite_semi_positive = np.isfinite(x).all(axis=1)
        m_pos = (x[finite_semi_positive] >= 0).all(axis=1)
        finite_semi_positive[finite_semi_positive] &= m_pos

    else:
        finite_semi_positive = np.isfinite(x) & (x >= 0)

    return np.asarray(finite_semi_positive)


def nans(shape):
    """Create a new numpy array filled with NaNs.

    :param shape: shape of the :class:`numpy.ndarray`
    :type shape: :obj:`int` or :obj:`tuple`

    :rtype: :class:`numpy.ndarray` (numpy.nan, dim=1)

    >>> from skpro.libs.cyclic_boosting.utils import nans
    >>> nans(3)
    array([ nan,  nan,  nan])

    >>> nans((2, 2))
    array([[ nan,  nan],
           [ nan,  nan]])
    """
    result = np.empty(shape)
    result.fill(np.nan)
    return result


def multidim_binnos_to_lexicographic_binnos(binnos, n_bins=None, binsteps=None):
    """Map index-tuples of ``M``-dimensional features to integers.

    In the cyclic boosting algorithm there is a one-dimensional array of
    factors for each one-dimensional feature. For processing of
    multi-dimensional variables (i.e. combinations of several one-dimensional
    feature variables) we have bins for all bin combinations. For example, if
    you have two one-dimensional features ``p`` and ``q`` with ``n_p`` and
    ``n_q`` bins, the two-dimensional feature ``(p, q)`` will have a
    two-dimensional factor array of shape ``(n_p, n_q)``.

    The internal representation of this array, however, is that of a
    one-dimensional array. Thus, a function is needed that maps index-tuples
    ``(p, q,...)`` to integer indices.  This is the purpose of this function.

    This function performs this mapping for ``N`` rows of index ``M``-tuples.

    If there are any missing values in a row, the returned ordinal number is
    set to ``np.prod(n_bins)``.

    Parameters
    ----------
    binnos: :class:`numpy.ndarray` of shape ``(N, M)``
        multidimensional bin numbers

    n_bins: :class:`numpy.ndarray` of shape ``(M,)`` or `None`
        number of bins in each dimension; if `None`, it will be determined from
        ``binnos``.

    binsteps: :class:`numpy.ndarray` of type `int64` and shape ``(M,)``
        bin steps as returned by :func:`bin_steps` when called on ``n_bins``;
        if `None`, it will be determined from ``binnos``.

    Returns
    -------
    tuple
        ordinal numbers of the bins in lexicographic order as
        :class:`numpy.ndarray` of type `int64` and maximal bin numbers as
        :class:`numpy.ndarray` of shape ``(M,)``

    Examples
    --------

    >>> binnos = np.c_[[1, 1, 0, 0, np.nan, 1, np.nan],
    ...                [0, 1, 2, 1, 1, 2, np.nan]]
    >>> n_bins = np.array([2, 3])
    >>> binsteps = bin_steps(n_bins)
    >>> from skpro.libs.cyclic_boosting.utils import multidim_binnos_to_lexicographic_binnos
    >>> lex_binnos, n_finite = multidim_binnos_to_lexicographic_binnos(
    ...     binnos, n_bins=n_bins, binsteps=binsteps)
    >>> lex_binnos
    array([3, 4, 2, 1, 6, 5, 6])
    >>> n_finite
    array([2, 3])
    >>> lex_binnos, n_finite =  multidim_binnos_to_lexicographic_binnos(binnos)
    >>> lex_binnos
    array([3, 4, 2, 1, 6, 5, 6])
    >>> n_finite
    array([2, 3])
    """
    if isinstance(binnos, pd.DataFrame):
        binnos = binnos.values

    if n_bins is None:
        finite_bins = binnos[slice_finite_semi_positive(binnos)]
        if len(finite_bins) == 0:
            n_bins = np.zeros(binnos.shape[1], dtype=np.int64)
        else:
            n_bins = np.max(finite_bins, axis=0).astype(np.int64) + 1

    if binsteps is None:
        binsteps = bin_steps(n_bins)

    is_valid = np.isfinite(binnos).all(axis=1)
    is_valid[is_valid] &= (
        (binnos[is_valid] >= 0) & (binnos[is_valid] < n_bins[None, :])
    ).all(axis=1)

    if not np.any(is_valid):
        result = np.repeat(0, len(binnos))
    else:
        result = np.repeat(np.prod(n_bins), len(binnos))

    result[is_valid] = np.dot(np.floor(binnos[is_valid]), binsteps[1:])

    if n_bins.prod() >= np.iinfo(np.int32).max:
        _logger.warning(
            "Enumerating multidimensional bins: A multidimensional feature of "
            "shape {n_bins} requires {total_bins} unique bins in total. "
            "Please consider whether this is what you want.".format(
                n_bins=n_bins, total_bins=n_bins.prod()
            )
        )
    result = result.astype(np.int64)
    return result, n_bins


@nb.njit()
def bin_steps(n_bins: nb.int64[:]):
    """
    Multidimensional bin steps for lexicographical order

    :param n_bins: number of bins for each dimension
    :type n_bins: ``numpy.ndarray`` (element-type ``float64``, shape ``(M,)``)

    :return: in slot ``i + 1`` for dimension ``i``, the number of steps needed
        for iterating through the following dimensions in lexicographical order
        until the bin number can be increased for dimension ``i``.

        See the doctests of :func:`arange_multi`: It's the number of rows between
        changes in column ``i``.

        In slot 0 this is the number steps needed to iterate through all
        dimensions.
    :rtype: ``numpy.ndarray`` (element-type ``int64``, shape ``(M + 1,)``)

    >>> from skpro.libs.cyclic_boosting.utils import bin_steps
    >>> bin_steps(np.array([3, 2]))
    array([6, 2, 1])

    >>> bin_steps(np.array([3, 2, 4, 1, 2]))
    array([48, 16,  8,  2,  2,  1])
    """
    total_bin_steps = 1
    M = n_bins.shape[0]
    bin_steps = np.empty(M + 1, dtype=np.int64)

    for m in range(M, -1, -1):
        bin_steps[m] = total_bin_steps
        total_bin_steps = n_bins[m - 1] * total_bin_steps

    return bin_steps


@nb.njit()
def arange_multi(stops) -> np.ndarray:
    """
    Multidimensional generalization of :func:`numpy.arange`

    :param stops: upper limits (exclusive) for the corresponding dimensions in
        the result
    :type stops: sequence of numbers

    :return: matrix of combinations of range values in lexicographic order
    :rtype: ``numpy.ndarray`` (element-type ``int64``, shape
        ``(n_bins, len(stops))``)

    >>> from skpro.libs.cyclic_boosting.utils import arange_multi
    >>> arange_multi([2, 3])
    array([[0, 0],
           [0, 1],
           [0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])

    In this example, the last dimension has only one bin:

    >>> arange_multi([3, 4, 1])
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 2, 0],
           [0, 3, 0],
           [1, 0, 0],
           [1, 1, 0],
           [1, 2, 0],
           [1, 3, 0],
           [2, 0, 0],
           [2, 1, 0],
           [2, 2, 0],
           [2, 3, 0]])

    >>> arange_multi([2])
    array([[0],
           [1]])
    """
    stops1 = np.asarray(stops)
    M = stops1.shape[0]
    bin_steps1 = bin_steps(stops1)
    total_bin_steps = bin_steps1[0]

    result = np.zeros((total_bin_steps, M), dtype=np.int64)

    for m in range(M):
        bin_steps_m = bin_steps1[m + 1]
        binno = 0
        for i in range(0, total_bin_steps, bin_steps_m):
            for j in range(bin_steps_m):
                result[i + j, m] = binno
            binno += 1
            if binno >= stops1[m]:
                binno = 0

    return result


def calc_linear_bins(x, nbins):
    """Calculate a linear binning for all values in x with nbins

    :param x: Input array.
    :type x: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)

    :param nbins: number of bins desired
    :type nbins: int

    **Example**

    >>> x = np.array([0.0, 0.1, 0.3, 0.4, 0.8, 0.9, 0.95, 1.1, 1.3, 1.5])
    >>> nbins = 3
    >>> from skpro.libs.cyclic_boosting.utils import calc_linear_bins
    >>> bin_boundaries, bin_centers = calc_linear_bins(x, nbins)
    >>> bin_centers
    array([ 0.25,  0.75,  1.25])
    >>> bin_boundaries
    array([ 0. ,  0.5,  1. ,  1.5])
    """
    minx, maxx = np.min(x), np.max(x)
    bin_boundaries = np.linspace(minx, maxx, nbins + 1)
    bin_centers = 0.5 * (bin_boundaries[1:] + bin_boundaries[:-1])
    return bin_boundaries, bin_centers


def digitize(x, bins):
    """
    This is an alternative version of :func:`numpy.digitize`. It puts x values
    greater or equal to bins.max() on the last index of `bins`.
    Values smaller than bins.min() are put on the first index of `bins`.

    :param x: Input array to be binned. It has to be 1-dimensional.
    :type x: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)

    :param bins: Array of bins. It has to be 1-dimensional and monotonic.
    :type bins: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)

    :return: Output array of indices, of same shape as `x`.
    :rtype: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)

    **Raises**

    ValueError
        If the input is not 1-dimensional, or if `bins` is not monotonic.
    TypeError
        If the type of the input is complex.

    **See Also**

    :func:`numpy.digitize`

    **Notes**

    If values in `x` are such that they fall outside the bin range,
    attempting to index `bins` with the indices that `digitize` returns
    will result in an IndexError.

    **Examples**

    >>> x = np.array([-1000, -0.2, 0.2, 6.4, 3.0, 10, 11, 1000])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> from skpro.libs.cyclic_boosting.utils import digitize
    >>> inds = digitize(x, bins)
    >>> inds
    array([0, 0, 1, 4, 3, 4, 4, 4])
    """
    bin_numbers = np.digitize(x, bins)
    is_max = x >= bins.max()
    bin_numbers[is_max] -= 1
    return bin_numbers


def calc_means_medians(binnumbers, y, weights=None):
    """Calculate the means, medians, counts, and errors for y grouped over the
    binnumbers.

    Parameters
    ----------

    binnumbers: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)
        binnumbers

    y: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)
        target values

    weights: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)
        array of event weights

    **Example**

    >>> binnumbers = np.array([0.0, 0., 0., 1., 1., 1., 2., 2., 3., 3.])
    >>> y = np.array([0.0, 0.2, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6])
    >>> from skpro.libs.cyclic_boosting.utils import calc_means_medians
    >>> means, medians, counts, errors = calc_means_medians(
    ...     binnumbers, y)
    >>> means
    0.0    0.233333
    1.0    0.716667
    2.0    1.100000
    3.0    1.500000
    dtype: float64

    >>> medians
    0.0    0.2
    1.0    0.7
    2.0    1.1
    3.0    1.5
    dtype: float64

    >>> errors
    [0.0   -0.13654
    1.0   -0.06827
    2.0   -0.06827
    3.0   -0.06827
    dtype: float64, 0.0    0.204810
    1.0    0.102405
    2.0    0.068270
    3.0    0.068270
    dtype: float64, 0.0   -0.19090
    1.0   -0.09545
    2.0   -0.09545
    3.0   -0.09545
    dtype: float64, 0.0    0.286350
    1.0    0.143175
    2.0    0.095450
    3.0    0.095450
    dtype: float64]

    >>> counts
    0.0    3
    1.0    3
    2.0    2
    3.0    2
    dtype: int64
    """
    with warnings.catch_warnings(record=False):
        warnings.simplefilter("ignore")
        if weights is None or len(np.unique(weights)) == 1:
            return _calc_means_medians_evenly_weighted(binnumbers, y)
        else:
            return _calc_means_medians_with_weights(binnumbers, y, weights)


def calc_weighted_quantile(binnumbers, y, weights, quantile):
    df = pd.DataFrame({"y": y, "weights": weights, "binnumbers": binnumbers})
    return df.groupby("binnumbers").apply(_weighted_quantile_of_dataframe(quantile))


def _calc_means_medians_evenly_weighted(binnumbers, y):
    """Calculate the means, medians, counts, and errors for y grouped over the
    binnumbers.

    Parameters
    ----------

    binnumbers: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)
        binnumbers

    y: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)
        target values
    """
    groupby = pd.Series(y).groupby(binnumbers)
    means = groupby.mean()
    medians = groupby.median()
    counts = groupby.size()

    quantiles = [
        groupby.quantile(LOWER_1_SIGMA_QUANTILE),
        groupby.quantile(1 - LOWER_1_SIGMA_QUANTILE),
        groupby.quantile(LOWER_2_SIGMA_QUANTILE),
        groupby.quantile(1 - LOWER_2_SIGMA_QUANTILE),
    ]

    errors = [series - medians for series in quantiles]

    return means, medians, counts, errors


def _calc_means_medians_with_weights(binnumbers, y, weights):
    """Calculate the means, medians, counts, and errors for y grouped over the
    binnumbers using weights.

    Parameters
    ----------

    binnumbers: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)
        binnumbers

    y: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)
        target values

    weights: :obj:`list` or :class:`numpy.ndarray` (float64, dim=1)
        array of event weights
    """
    df = pd.DataFrame({"y": y, "weights": weights, "binnumbers": binnumbers})
    groupby = df.groupby("binnumbers")

    means = groupby.apply(_weighted_mean_of_dataframe)
    medians = groupby.apply(_weighted_median_of_dataframe)
    counts = groupby.apply(_weighted_size_of_dataframe)

    quantiles = [
        groupby.apply(_weighted_quantile_of_dataframe(LOWER_1_SIGMA_QUANTILE)),
        groupby.apply(_weighted_quantile_of_dataframe(1 - LOWER_1_SIGMA_QUANTILE)),
        groupby.apply(_weighted_quantile_of_dataframe(LOWER_2_SIGMA_QUANTILE)),
        groupby.apply(_weighted_quantile_of_dataframe(1 - LOWER_2_SIGMA_QUANTILE)),
    ]
    errors = [series - medians for series in quantiles]

    return means, medians, counts, errors


def _weighted_quantile_of_dataframe(quantile):
    """
    Generates a function which calculates the weighted quantile of a given pandas.DataFrame

    Parameters
    ----------

    quantile: float between 0 and 1
    """

    def quantile_calculation(x):
        """
        Calculates the weighted quantile of a given pandas.DataFrame.
        Expects two columns: y and weights

        Parameters
        ----------

        x: :class:`pandas.DataFrame` with (at least) two columns y and weights
        """
        x = x.sort_values("y")
        cumsum = x.weights.cumsum()
        cutoff = x.weights.sum() * quantile
        try:
            return x.y[cumsum >= cutoff].iloc[0]
        except IndexError:
            return min(x.y)

    return quantile_calculation


def _weighted_median_of_dataframe(x):
    """
    Calculates the weighted size of a given pandas.DataFrame.
    Expects two columns: y and weights

    Parameters
    ----------

    x: :class:`pandas.DataFrame` with (at least) two columns y and weights
    """
    x = x.sort_values("y")
    cumsum = x.weights.cumsum()
    cutoff = x.weights.sum() / 2.0
    try:
        return x.y[cumsum >= cutoff].iloc[0]
    except IndexError:
        return min(x.y)


def _weighted_mean_of_dataframe(x):
    """
    Calculates the weighted mean of a given pandas.DataFrame.
    Expects two columns: y and weights

    Parameters
    ----------

    x: :class:`pandas.DataFrame` with (at least) two columns y and weights
    """
    weighted_sum = (x.y * x.weights).sum()
    sum_of_weights = x.weights.sum()
    return weighted_sum / sum_of_weights


def _weighted_size_of_dataframe(x):
    """
    Calculates the weighted size of a given pandas.DataFrame.
    Expects one column: weights

    Parameters
    ----------

    x: :class:`pandas.DataFrame` with (at least) one column called weights
    """
    return x.weights.sum()


def weighted_stddev(values, weights):
    r"""Calculates the weighted standard deviation :math:`\sigma`:

    .. math::

        \sigma = \frac{\sum_i w_i \cdot (x_i - /mu_{wx})^2}{\sum_i w_i}

    Parameters
    ----------
    values: :class:`numpy.ndarray` (float64, dim=1)
        Values of the samples :math:`x_i`.
    weights: :class:`numpy.ndarray` (float64, dim=1)
        Weights of the samples :math:`w_i`.
    """
    mean_w = np.sum(weights * values) / np.sum(weights)
    numerator = np.sum(weights * (values - mean_w) ** 2)
    denominator = np.sum(weights)
    stddev_w = np.sqrt(numerator / denominator)
    return stddev_w


def clone(estimator, safe=True):
    """Constructs a new estimator with the same constructor parameters.

    A better name for this function would be 'reconstruct'.

    This is a reimplemenation of :func:`sklearn.base.clone` respecting wishes of
    estimators and their subestimators to avoid reconstructing certain
    attributes.

    Clone performs a deep copy of the model in an estimator
    without actually copying attached data. It yields a new estimator
    with the same parameters that has not been fit on any data.
    """
    estimator_type = type(estimator)
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, "get_params"):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            raise TypeError(
                "Cannot clone object '%s' (type %s): "
                "it does not seem to be a scikit-learn estimator as "
                "it does not implement a 'get_params' methods."
                % (repr(estimator), type(estimator))
            )
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)

    if hasattr(estimator, "no_deepcopy"):
        no_deepcopy = estimator.no_deepcopy
        for name, param in new_object_params.items():
            if name not in no_deepcopy:
                new_object_params[name] = clone(param, safe=False)
    else:
        for name, param in new_object_params.items():
            new_object_params[name] = clone(param, safe=False)

    new_object = klass(**new_object_params)

    return new_object


def regularize_to_prior_expectation(
    values, uncertainties, prior_expectation, threshold=2.5
):
    r"""Regularize values with uncertainties to a prior expectation.

    :param values: measured values
    :type values: :class:`numpy.ndarray` (float64, dim=1)

    :param uncertainties: uncertainties for the values.
    :type uncertainties: :class:`numpy.ndarray` (float64, dim=1)

    :param prior_expectation: The prior expectations dominate
        the regularized value if the uncertainties are large.
    :type prior_expectation: :class:`numpy.ndarray` (float64, dim=1) or float

    :param threshold: Threshold in terms of sigma. If the significance of a
        value:

        .. math::

            \text{sig}(x_i) = \frac{x_i - \text{prior\_expectation}_i}
            {\text{uncertainty}_i}

        is below the threshold, the prior expectation replaces the value.
    :type threshold: float

    **Doctests**

    >>> values = np.array([0, 1, 2, 3])
    >>> uncertainties = np.ones_like(values)*0.1
    >>> prior_expectation = 1.
    >>> from skpro.libs.cyclic_boosting.utils import regularize_to_prior_expectation
    >>> regularized_values = regularize_to_prior_expectation(
    ... values, uncertainties, prior_expectation)
    >>> regularized_values
    array([ 0.03175416,  1.        ,  1.96824584,  2.98431348])
    >>> np.allclose(1 - np.sqrt(((values[0] - 1) / 0.1)**2 - 2.5**2) * 0.1,
    ...     regularized_values[0])
    True
    >>> np.allclose(1 + np.sqrt(((values[-1] - 1) / 0.1)**2 - 2.5**2) * 0.1,
    ...     regularized_values[-1])
    True
    """
    significance = (values - prior_expectation) / uncertainties
    return prior_expectation + uncertainties * np.where(
        np.abs(significance) > threshold,
        np.sign(significance) * np.sqrt(np.abs(significance**2.0 - threshold**2.0)),
        0,
    )


def regularize_to_error_weighted_mean(values, uncertainties, prior_prediction=None):
    r"""Regularize values with uncertainties to its error-weighted mean.

    :param values: measured values
    :type values: :class:`numpy.ndarray` (float64, dim=1)

    :param uncertainties: uncertainties for the values.
    :type uncertainties: :class:`numpy.ndarray` (float64, dim=1)

    :param prior_prediction: If the `prior_prediction` is specified, all values
        are regularized with it and not with the error weighted mean.
    :type prior_prediction: float

    :returns: regularized values
    :rtype: :class:`numpy.ndarray` (float64, dim=1)

    The error weighted mean is defined as:

    .. math::

         \bar{x}_{\sigma} = \frac{\sum\limits_{i} \frac{1}
         {\sigma^2_i} \cdot x_i}{\sum\limits_{i} \frac{1}{\sigma^2_i}}

    with the uncertainties :math:`\sigma_i` and values :math:`x_i`.
    The uncertainty :math:`\sigma_{\bar{x}}` for :math:`\bar{x}`
    is calculated as:

    .. math::

        \sigma^2_{\bar{x}} = \frac{\sum\limits_{i} \frac{1}{\sigma^2_i}
        (x - \bar{x}_{\sigma})^2}{\sum\limits_{i} \frac{1}{\sigma^2_i}}

    The regularized values are calculated as follows:

    .. math::

        x_i^{'} = \frac{\frac{1}{\sigma^2_i} \cdot x_i +
        \frac{1}{\sigma^2_{\bar{x}}} \cdot \bar{x}}
        {\frac{1}{\sigma^2_i} + \frac{1}{\sigma^2_{\bar{x}}}}

    >>> values = np.array([100., 100., 100., 90., 110., 180., 180., 20., 20.])
    >>> uncertainties = np.array([10., 10., 10., 10., 10., 10., 15., 10., 15.])
    >>> from skpro.libs.cyclic_boosting.utils import regularize_to_error_weighted_mean
    >>> regularize_to_error_weighted_mean(values, uncertainties)
    array([ 100.        ,  100.        ,  100.        ,   90.40501997,
            109.59498003,  176.75984027,  173.06094747,   23.24015973,
             26.93905253])

    >>> values = np.array([100.])
    >>> uncertainties = np.array([10.])
    >>> np.allclose(regularize_to_error_weighted_mean(values, uncertainties),
    ...             values)
    True

    >>> values = np.array([100., 101.])
    >>> uncertainties = np.array([10., 10.])
    >>> regularize_to_error_weighted_mean(
    ...     values, uncertainties, prior_prediction=50.)
    array([ 98.11356348,  99.07583475])

    >>> values = np.array([100., 10])
    >>> uncertainties = np.array([10.])
    >>> regularize_to_error_weighted_mean(values, uncertainties)
    Traceback (most recent call last):
    ValueError: <values> and <uncertainties> must have the same shape
    """
    if values.shape != uncertainties.shape:
        raise ValueError("values and uncertainties must have the same shape")
    if len(values) < 1 or (prior_prediction is None and len(values) == 1):
        return values

    x = values
    wx = 1.0 / np.square(uncertainties)
    sum_wx = np.sum(wx)
    if prior_prediction is None:
        if np.allclose(x, x[0]):
            # if all values are the same,
            # regularizing to the mean makes no sense
            return x
        x_mean = np.sum(wx * x) / sum_wx
    else:
        if np.allclose(x, prior_prediction):
            return x
        x_mean = prior_prediction
    wx_incl = 1.0 / (np.sum(wx * np.square(x - x_mean)) / sum_wx)
    res = (wx * x + wx_incl * x_mean) / (wx + wx_incl)
    return res


def regularize_to_error_weighted_mean_neighbors(
    values: np.ndarray, uncertainties: np.ndarray, window_size: Optional[int] = 3
) -> np.ndarray:
    """
    Regularize values with uncertainties to its error-weighted mean, using a
    sliding window.

    Parameters
    ----------
    values : np.ndarray
        data (`float` type) to be regularized
    uncertainties : np.ndarray
        uncertainties (`float` type) of values
    window_size : int
        size of the sliding window to be used (e.g., 3 means include direct
        left and right neighbors)

    Returns
    -------
    np.ndarray
        regularized values
    """
    if values.shape != uncertainties.shape:
        raise ValueError("values and uncertainties must have the same shape")

    if len(values) < 3:
        return regularize_to_error_weighted_mean(values, uncertainties)

    window_arr = np.ones(window_size)
    x = values
    wx = np.where(uncertainties > 0, 1.0 / np.square(uncertainties), 0.0)

    sum_wx = np.convolve(wx, window_arr, "same")
    x_mean = np.convolve(wx * x, window_arr, "same") / sum_wx

    wx_incl = np.ones(len(x))
    for i in range(len(x)):
        wx_incl[i] = (
            sum_wx / np.convolve(wx * np.square(x - x_mean[i]), window_arr, "same")
        )[i]

    res = (wx * x + wx_incl * x_mean) / (wx + wx_incl)
    return res


def neutralize_one_dim_influence(values, uncertainties):
    """
    Neutralize influence of one dimensional features in a two-dimensional factor matrix

    :param values: measured values
    :type values: :class:`numpy.ndarray` (float64, dim=2)

    :param uncertainties: uncertainties for the values.
    :type uncertainties: :class:`numpy.ndarray` (float64, dim=2)

    returns an updated 2D-table that has no net shift on any of the 2 projections
    """
    deviation = 100.0
    iteration = 0
    T0 = np.einsum("ij->i", uncertainties)
    T1 = np.einsum("ij->j", uncertainties)
    S0 = np.einsum("ij->i", values * uncertainties)

    while deviation > 0.01 and iteration < 4:
        iteration += 1

        #  row corrections
        values = values - np.outer(S0 / T0, np.ones(values.shape[1]))

        # recalc column sum after row correction
        S1 = np.sum(values * uncertainties, axis=0)

        #  column corrections
        values = values - np.outer(np.ones(values.shape[0]), S1 / T1)

        # recalc column and row sums after column correction
        S0 = np.sum(values * uncertainties, axis=1)
        S1 = np.sum(values * uncertainties, axis=0)

        #  check total absolute row and column sum for determining convergence
        deviation = abs(S0).sum() + abs(S1).sum()
    return values


def get_bin_bounds(binners, feat_group):
    """
    Gets the bin boundaries for each feature group.

    Parameters
    ----------
    binners: list
        List of binners.
    feat_group: str or tuple of str
        A feature property for which the bin boundaries should be extracted
        from the binners.
    """
    if binners is None:
        return None
    bin_bounds = {}
    for binner in binners:
        bin_bounds.update(binner.get_feature_bin_boundaries())
    if feat_group in bin_bounds and bin_bounds[feat_group] is not None:
        return bin_bounds[feat_group][:, 0]
    else:
        return None


def generator_to_decorator(gen):
    """Turn a generator into a decorator.

    The mechanism is similar to :func:`contextlib.contextmanager` which turns
    a generator into a contextmanager. :mod:`decorator` is used internally.

    Thanks to :mod:`decorator`, this function preserves the docstring and the
    signature of the function to be decorated.

    The docstring of the resulting decorator will include the original
    docstring of the generator and an additional remark stating
    that this is the corresponding decorator.

    :param gen: wrapping the function call.
    :type gen: generator function

    :return: decorator

    For a detailed example, see
    :func:`generator_to_decorator_and_contextmanager`.
    """

    @decorator.decorator
    def created_decorator(func, *args, **kwargs):
        gen_instance = gen()
        try:
            next(gen_instance)
            return func(*args, **kwargs)
        finally:
            gen_instance.close()

    doc = gen.__doc__ or ""
    created_decorator.__doc__ = doc + "\n    This is the corresponding decorator."
    return created_decorator


def linear_regression(x, y, w):
    r"""Performs a linear regression allowing uncertainties in y.

    :math:`f(x) = y = \alpha + \beta \cdot x`

    :param x: x vector
    :type x: :class:`numpy.ndarray`

    :param y: y vector
    :type y: :class:`numpy.ndarray`

    :param w: weight vector
    :type w: :class:`numpy.ndarray`

    :returns: The coefficients `alpha` and `beta`.
    :rtype: :obj:`tuple` of 2 :obj:`float`
    """
    s_w = np.sum(w)
    s_wx = np.dot(w, x)
    s_wxx = np.dot(np.square(x), w)
    s_wy = np.dot(w, y)
    s_wxy = np.sum(w * x * y)
    det = s_w * s_wxx - s_wx * s_wx
    alpha = (s_wxx * s_wy - s_wx * s_wxy) / det
    beta = (s_w * s_wxy - s_wx * s_wy) / det
    return alpha, beta


def get_feature_column_names(X, exclude_columns=[]):
    features = list(X.columns)
    for col in exclude_columns:
        if col in features:
            features.remove(col)
    return features


def continuous_quantile_from_discrete_pdf(y, quantile, weights):
    """
    Calculates a continous quantile value approximation for a given quantile
    from an array of potentially discrete values.

    Parameters
    ----------
    y : np.ndarray
        containing data with `float` type (potentially discrete)
    quantile : float
        desired quantile

    Returns
    -------
    float
        calculated quantile value
    """
    y = np.asarray(y)
    weights = np.asarray(weights)

    sorting = y.argsort()
    sorted_y = y[sorting]
    cumsum = weights[sorting].cumsum()
    quantile_index = weights.sum() * quantile
    quantile_y = sorted_y[cumsum >= quantile_index][0]

    all_quantile_y = np.where(sorted_y == quantile_y)[0]
    index_low = all_quantile_y[0]
    index_high = all_quantile_y[-1]
    if index_high > index_low:
        quantile_y += (int(quantile_index) - index_low) / (index_high - index_low)

    return quantile_y


@dataclass
class ConvergenceParameters:
    """Class for registering the convergence parameters"""

    loss_change: float = 1e20
    delta: float = 100.0

    def set_loss_change(self, updated_loss_change: float) -> None:
        self.loss_change = updated_loss_change

    def set_delta(self, updated_delta: float) -> None:
        self.delta = updated_delta


def get_normalized_values(values: Iterable) -> List[float]:
    values_total = sum(values)
    if round(values_total, 6) != 0.0:
        return [value / values_total for value in values]
    return [value for value in values]


def smear_discrete_cdftruth(cdf_func: callable, y: int) -> float:
    """
    Smearing of the CDF value of a sample from a discrete random variable. Main
    usage is for a histogram of CDF values to check an estimated individual
    probability distribution (should be flat).

    Parameters
    ----------
    y : int
        value from discrete random variable
    cdf_func : callable
        cumulative distribution function

    Returns
    -------
    float
        smeared CDF value for y
    """
    cdf_high = cdf_func(y)
    if y >= 1:
        cdf_low = cdf_func(y - 1)
    else:
        cdf_low = 0.0
    cdf_truth = cdf_low + np.random.uniform(0, 1) * (cdf_high - cdf_low)
    return cdf_truth


def smear_discrete_cdftruth_qpd(qpds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Smearing of the CDF values of samples from discrete random variables. Main
    usage is for a histogram of CDF values to check an estimated individual
    probability distribution (should be flat).

    Parameters
    ----------
    y : np.ndarray
        values from discrete random variables
    qpds : np.ndarray
        array of QPDs

    Returns
    -------
    np.ndarray
        smeared CDF values for y
    """
    cdf_high = qpds.cdf(y, inner=True)
    cdf_low = np.where(y >= 1, qpds.cdf(y - 1, inner=True), 0.0)
    cdf_truth = cdf_low + np.random.uniform(0, 1, len(y)) * (cdf_high - cdf_low)
    return cdf_truth
