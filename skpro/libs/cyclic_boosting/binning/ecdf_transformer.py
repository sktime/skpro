import logging
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import sklearn.base as sklearnb

from skpro.libs.cyclic_boosting import flags

from ._binary_search import eq_multi, ge_lim, le_interp_multi
from ._utils import (
    _read_feature_property,
    check_frame_empty,
    get_column_index,
    minimal_difference,
)

_logger = logging.getLogger(__name__)


class ConstFunction:
    def __init__(self, val):
        self.val = val

    def __call__(self):
        return self.val


class ECdfTransformer(sklearnb.BaseEstimator, sklearnb.TransformerMixin):
    r"""Transform features to the empirical CDF scale of the training data.

    CDF = :math:`P\left(X \leq x\right)` = cumulative distribution function.
    See `CDF on wikipedia
    <http://en.wikipedia.org/wiki/Cumulative_distribution_function>`_

    Each feature found in ``feature_properties`` is considered in separation.

    In :meth:`fit`, (up to) ``n_bins`` bin boundaries with approximately equal
    number of data points are determined.  For discrete values, the complete CDF
    is stored and ``n_bins`` is ignored.

    In :meth:`transform`, each feature value is associated with the
    corresponding bin by binary search. For features with
    :obj:`cyclic_boosting.flags.IS_CONTINUOUS` set the empirical CDF is then
    interpolated between the left and the right bin boundary.  For out-of-range
    features, the bin boundaries are taken.  For features with
    :obj:`cyclic_boosting.flags.IS_ORDERED` or
    :obj:`cyclic_boosting.flags.IS_UNORDERED` only values that have been seen
    in the fit are transformed to the corresponding empirical CDF values. For
    all values, not within epsilon of the values seen in the fit, `numpy.nan`
    is returned. Missing values(:obj:`numpy.nan`) stay missing values and are
    not transformed regardless of the ``feature_properties`` set and feature
    values seen in :meth:`fit`. For all features the feature property
    :obj:`cyclic_boosting.flags.HAS_MISSING` is assumed.

    Parameters
    ----------

    n_bins: int, dict
        Maximum number of bins used to estimate the empirical CDF.  ``n_bins``
        is ignored for features with discrete preprocessing.
        If a dict is passed, the feature names/indices should be the keys and the
        n_bins are the values. Example : ``{'feature a': 150, 'feature b': 20}``

    feature_properties: dict
        Dictionary listing the names of all features as keys and their
        preprocessing flags as values. When using a numpy feature matrix X with
        no column names the keys of the feature properties are the column
        indices.  If no ``feature_properties`` are passed, all columns in ``X``
        are treated as `cyclic_boosting.flags.IS_CONTINUOUS`.  For more
        information about feature properties:

        .. seealso::
            :mod:`cyclic_boosting.flags`

    weight_column
        Optional column label or column index for the weight column.  If not set
        all samples receive the same weight 1.

    epsilon: float
        Used thresholds for the comparison of float values:

         * ``epsilon * 1.0`` for the comparison of CDF values
         * ``epsilon * minimal_bin_width`` for the comparison with bin
           boundaries of a given feature

        Default value for epsilon: 1e-9

    tolerance: double
        Relative tolerance of the minimum bin weight. (E.g.
        if you specify 100 bins and a tolerance of 0.05 the bins are
        required to have only 0.95% of the total bin weights instead of
        1.0%)


    **Guarantees for continuous features**
    (cyclic_boosting.flags.IS_CONTINUOUS set for feature)

    * The estimated number of bins :math:`n_\text{bins\_estimated}` is always
      smaller equal than the number of bins
      requested by the user :math:`n_\text{bins}`.

      .. math::
          n_\text{bins\_estimated} \leq n_\text{bins}

    * The bin boundaries are chosen such that each bin contains at least
      a fraction of :math:`\frac{1}{n_\text{bins}}` of all values.

    **Guarantees for discrete features**
    (flags.UNORDERED or flags.ORDERED set for feature)

    * The estimated number of bins :math:`n_\text{bins\_estimated}` is equal
      to the number of unique values :math:`n_\text{unique\_values}` found.

      .. math::
           n_\text{bins\_estimated} \Leftrightarrow n_\text{unique\_values}

    **Estimated parameters**

    Attributes
    ----------
    bins_and_cdfs_
        For each feature, a tuple containing

         * the column name or index
         * the epsilon used for comparisons to bin boundaries; it is the
           constructor parameter ``epsilon`` multiplied by the smallest bin
           width

         * and the :class:`numpy.ndarray` of shape ``(at most n_bins + 1, 2)``

           This is a matrix containing the **bin boundaries** (column 0) and
           the **corresponding cumulative probabilities** (column 1) is
           learned in the fit. The matrix looks for one feature ``x`` like this:

           .. math::
              \begin{pmatrix}
              x_\text{min} & P\left(X < x_\text{min}\right) = 0 \\
              x_\text{boundary1} & P\left(X \leq x_\text{boundary1}\right) \\
              x_\text{boundary2} & P\left(X \leq x_\text{boundary2}\right) \\
              \ldots & \ldots \\
              x_\text{max} & P\left(X \leq x_\text{max}\right) = 1 \\
              \end{pmatrix}

           For mixed discrete and continuous features, there might be fewer than
           ``n_bins`` bins. For discrete features ``n_bins`` is ignored and
           the ``cdf`` is calculated for each unique value.
           type of `bins_and_cdfs_`: item :obj:`list` of :obj:`tuple`

    Examples
    --------

    >>> feature_1 = np.asarray([2.1, 2.2, 2.5, 3.1, 3.3, 3.7, 4.1, 4.4])
    >>> X = np.c_[feature_1]
    >>> eps = 1e-8

    >>> from skpro.libs.cyclic_boosting.binning import ECdfTransformer
    >>> trans = ECdfTransformer(n_bins=4, epsilon=eps)
    >>> trans = trans.fit(X)

    >>> # only one input column
    >>> column, epsilon, bins_cdfs = trans.bins_and_cdfs_[0]
    >>> assert column == 0 and np.allclose(epsilon, eps * 0.1)
    >>> bins_cdfs
    array([[ 2.1 ,  0.  ],
           [ 2.2 ,  0.25],
           [ 3.1 ,  0.5 ],
           [ 3.7 ,  0.75],
           [ 4.4 ,  1.  ]])

    >>> X_test = np.c_[[1.9, 2.4, 2.2, 3.6, 3.5, 4.3, 5.1]]
    >>> trans.transform(X_test)
    array([[ 0.        ],
           [ 0.30555556],
           [ 0.25      ],
           [ 0.70833333],
           [ 0.66666667],
           [ 0.96428571],
           [ 1.        ]])
    """

    def __init__(
        self,
        n_bins=100,
        feature_properties=None,
        weight_column=None,
        epsilon=1e-9,
        tolerance=0.1,
    ):
        self.n_bins = n_bins
        self.feature_properties = feature_properties
        self.weight_column = weight_column
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.bins_and_cdfs_ = None

    @staticmethod
    def _normalize_bins(n_bins):
        if isinstance(n_bins, int):
            return defaultdict(ConstFunction(n_bins))
        else:
            return n_bins

    def fit(self, X, y=None):
        self._nbins_per_feature = self._normalize_bins(self.n_bins)
        self.bins_and_cdfs_ = []

        if check_frame_empty(X):
            raise ValueError("Your input matrix for the binning is empty.")

        feature_columns = get_feature_column_names_or_indices(
            X, exclude_columns=[self.weight_column]
        )
        weights = get_weight_column(X, self.weight_column)

        for col in feature_columns:
            _logger.info(f"{self.__class__.__name__} column: {col}")
            x_col = get_X_column(X, col)

            feature_prop = _read_feature_property(col, self.feature_properties)

            if feature_prop is None:
                continue

            bins_x, cdf_x, _wsum, _n_nan = calculate_cdf_from_weighted_data(
                x_col.astype(float), weights
            )

            if len(bins_x) == 0 or len(cdf_x) == 0:
                self.bins_and_cdfs_.append((col, self.epsilon, None))
                continue

            if flags.is_ordered_set(feature_prop) or flags.is_unordered_set(
                feature_prop
            ):
                bin_boundaries = np.r_[bins_x[0], bins_x]
                cdf = np.r_[0.0, cdf_x]
            else:
                bin_boundaries, cdf = reduce_cdf_and_boundaries_to_nbins(
                    bins_x,
                    cdf_x,
                    self._nbins_per_feature[col],
                    self.epsilon,
                    self.tolerance,
                )

            n = len(cdf)
            bins_and_cdfs = np.empty((n, 2))
            bins_and_cdfs[:, 0] = bin_boundaries
            bins_and_cdfs[:, 1] = cdf

            epsilon = self.epsilon * minimal_difference(bin_boundaries)

            self.bins_and_cdfs_.append((col, epsilon, bins_and_cdfs))
        return self

    def _check_input_for_transform(self, X):
        if self.bins_and_cdfs_ is None:
            raise RuntimeError("Fit was not called before.")

        columns = get_feature_column_names_or_indices(
            X, exclude_columns=[self.weight_column]
        )
        if self.feature_properties is not None:
            columns = [col for col in columns if col in self.feature_properties]
        n_cols = len(columns)
        if n_cols != len(self.bins_and_cdfs_):
            raise ValueError(
                "Input Matrix X has not the same number"
                " of feature columns (%s) as "
                "the matrix in the fit (%s)." % (n_cols, len(self.bins_and_cdfs_))
            )

    def transform(self, X, y=None):
        self._check_input_for_transform(X)

        if check_frame_empty(X):
            return X

        Xnp = np.asarray(X, dtype=float)
        Xt = Xnp

        for col, epsilon, bins_and_cdfs in self.bins_and_cdfs_:
            j = get_column_index(X, col)
            feature_property = _read_feature_property(col, self.feature_properties)

            if feature_property is None:
                continue

            if bins_and_cdfs is not None:
                if flags.is_continuous_set(feature_property):
                    le_interp_multi(
                        bins_and_cdfs[:, 0],
                        Xnp[:, j],
                        bins_and_cdfs[:, 1],
                        0.0,
                        epsilon,
                        Xt[:, j],
                    )
                else:
                    eq_multi(
                        bins_and_cdfs[:, 0],
                        Xnp[:, j],
                        bins_and_cdfs[:, 1],
                        epsilon,
                        Xt[:, j],
                    )

            elif bins_and_cdfs is None:
                Xnp[:, j] = np.nan

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(Xt, columns=X.columns)
        else:
            return Xt


def get_feature_column_names_or_indices(
    X: Union[pd.DataFrame, np.ndarray],
    exclude_columns: Optional[Union[List[str], List[int]]] = None,
) -> Union[List[str], List[int]]:
    """
    Extract the column names from `X`. If `X` is a numpy matrix
    each column is labeled with an integer starting from zero.

    :param X: input matrix
    :type X: numpy.ndarray(dim=2) or pandas.DataFrame

    :param exclude_columns: column names or indices to omit.
    :type exclude_columns: list of int or str

    :rtype: list

    >>> X = np.c_[[0, 1], [1,0], [3, 5]]
    >>> from skpro.libs.cyclic_boosting.binning import get_feature_column_names_or_indices
    >>> get_feature_column_names_or_indices(X)
    [0, 1, 2]

    >>> get_feature_column_names_or_indices(X, exclude_columns=[1])
    [0, 2]

    >>> get_feature_column_names_or_indices(X, exclude_columns=[1, 1])
    [0, 2]

    >>> get_feature_column_names_or_indices(X, exclude_columns=[0, 1, 2])
    []

    >>> X = pd.DataFrame(X, columns = ['b', 'c', 'a'])
    >>> get_feature_column_names_or_indices(X, exclude_columns=['a'])
    ['b', 'c']

    >>> get_feature_column_names_or_indices(X, exclude_columns=['d'])
    ['b', 'c', 'a']
    """
    if isinstance(X, pd.DataFrame):
        columns = list(X.columns)
    elif isinstance(X, np.ndarray):
        assert X.ndim == 2, "X must be a 2D matrix"
        columns = list(range(0, X.shape[1]))
    else:
        raise ValueError("X must be a pandas.DataFrame or a numpy.ndarray")

    if exclude_columns is not None:
        exclude_columns = set(exclude_columns)
        return [x for x in columns if x not in exclude_columns]
    else:
        return columns


def get_weight_column(X, weight_column=None):
    """
    Check if a weight column is present and return it if
    possible. If no weight columns is present in `X` a
    weight column with only ``ones`` of same length than
    `X` is created and returned.

    :param X: Samples feature matrix.
    :type X: numpy.ndarray(dim=2) or pandas.DataFrame

    :param weight_column: Name or index of the weight column or None.
    :type weight_column: int or string or ``NoneType``

    :rtype: numpy.ndarray

    >>> X = np.c_[[0., 1], [1,0], [3, 5]]
    >>> from skpro.libs.cyclic_boosting.binning import get_weight_column
    >>> get_weight_column(X)
    array([ 1.,  1.])
    >>> get_weight_column(X, 0)
    array([ 0.,  1.])
    >>> get_weight_column(X, 2)
    array([ 3.,  5.])

    >>> X = pd.DataFrame(X, columns = ['b', 'c', 'a'])
    >>> get_weight_column(X)
    array([ 1.,  1.])

    >>> get_weight_column(X, 'c')
    array([ 1.,  0.])
    """
    if weight_column is not None:
        if isinstance(X, pd.DataFrame):
            try:
                return np.asarray(X[weight_column])
            except:
                raise ValueError(f"Weight column {str(weight_column)} not found in X.")
        else:
            try:
                return X[:, weight_column]
            except:
                raise ValueError(
                    f"Index {str(weight_column)} defining weight column not found in X."
                )
    else:
        return np.ones(X.shape[0], dtype=np.float64)


def reduce_cdf_and_boundaries_to_nbins(bins_x, cdf_x, n_bins, epsilon, tolerance):
    """
    Section the cdf spectrum into `n_bin` parts of equal statistics, and find
    all events beloning into these bins by filtering all suitable events in
    the event-wise `cdf_x` array.

    Often, events cannot be distributed exactly with equal statistics over all
    bins, therefore the ``tolerance`` argument allows for bins to be of a weight
    below  1.0 / n_bins.

    A minimum weight of 1.0 / n_bins - tolerance per bin is guaranteed.

    This function is used internally in the method
    :meth:`cyclic_boosting.binning.ECdfTransformer`.

    Parameters
    ----------
    bins_x: np.ndarray
        strictly increasing array containing all bin boundaries, length is the
        number of evenets.

    cdf_x: np.ndarray
        Strictly increasing array containing the cdf values corresponding to the
        bin boundaries in `bin_x`.  Contains one value for each event.

    n_bins: int
        Maximum number of bins that ought to be returned. This also determines
        the minimum weight per bin, which is 1 / n_bins.

    epsilon: double
        Threshold for the comparison of CDFs

    tolerance: double
        Relative tolerance of the minimum bin weight. (E.g.
        if you specify 100 bins and a tolerance of 0.05 the bins are
        required to have only 0.95% of the total bin weights instead of
        1.0%)

    Returns
    -------
    The ``reduced`` input arrays `bins_x` and `cdf_x`, now with **maximum**
    length n_bins, tuple of numpy.ndarrays(dim=1)
    """
    if n_bins < 2:
        raise ValueError("N_bins = %s has to greater than 1!", n_bins)

    n_cdf = n_bins + 1

    bin_boundaries = np.zeros(n_cdf, dtype=np.float64)
    cdf = np.zeros(n_cdf, dtype=np.float64)

    bin_boundaries[0] = bins_x[0]

    n = cdf_x.shape[0]
    index = 0
    cdf_share = 1.0 / n_bins
    previous_cdf = 0.0

    for i in range(1, n_bins + 1):
        cdf_rest = previous_cdf % cdf_share
        cdf_searched = previous_cdf + cdf_share

        if cdf_rest <= tolerance * cdf_share:
            cdf_searched -= cdf_rest

        if cdf_searched <= 1.0 + tolerance * cdf_share:
            index = ge_lim(cdf_x, cdf_searched - epsilon, 1, index, n)

            previous_cdf = cdf_x[index]
            cdf[i] = cdf_x[index]
            bin_boundaries[i] = bins_x[index]
        else:
            cdf[i - 1] = 1.0
            bin_boundaries[i - 1] = bins_x[n - 1]
            n_cdf = i
            break
    return np.array(bin_boundaries[:n_cdf]), np.array(cdf[:n_cdf])


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


def calculate_cdf_from_weighted_data(z, w):
    """
    Calculate the cdf value for each unique value in `z` weighted with the
    sample weights in `w`. All values not finite values in `z`
    and unique values of z with weight zero are ignored.

    Parameters
    ----------
    z: numpy.ndarray of float64
        input array

    w: numpy.ndarray
        sample weights


    Returns
    -------
    tuple of two :class:`numpy.ndarray`, a double and an int
        Tuple consisting of an array containing the valid unique `z`
        values, an array containing the cdf values for the valid `z` values,
        the total weight sum and the number of non finite values in `z`.

    Examples
    --------
    >>> z = np.array([1., 2., 3., 4., 5., 6., np.nan, 6.])
    >>> w = np.array([4., 2., 2., 1., 0., 1., 1.,     0.])
    >>> z_unique, cdfs, wsum, n_nan = calculate_cdf_from_weighted_data(z, w)
    >>> wsum
    10.0
    >>> n_nan
    1
    >>> z_unique  # array of unique values of z
    array([ 1.,  2.,  3.,  4.,  6.])
    >>> cdfs  # corresponding cdf values to z_unique
    array([ 0.4,  0.6,  0.8,  0.9,  1. ])
    """
    if z.shape[0] != w.shape[0]:
        raise ValueError("input vectors must be of same shape")

    n_nan = np.count_nonzero(np.isnan(z))
    z_unique = np.unique(z[~np.isnan(z)])

    wsum = np.nansum(w[~np.isnan(z)])

    # Accumulate the weights for the unique values.
    uniques = (
        pd.DataFrame({"z": z, "w": w}).groupby(["z"]).agg({"w": "sum"}).reset_index()
    )
    uniques = uniques.loc[uniques["w"] != 0]

    cdf = np.nancumsum(uniques["w"]) / wsum

    return z_unique, cdf, wsum, n_nan
