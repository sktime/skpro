import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from skpro.libs.cyclic_boosting import flags

from ._binary_search import eq_multi, ge_multi
from ._utils import _read_feature_property, check_frame_empty
from .ecdf_transformer import ECdfTransformer

MISSING_VALUE_AS_BINNO = -1

_logger = logging.getLogger(__name__)


class BinNumberTransformer(ECdfTransformer):
    r"""This transformer bins feature-variables in ``X`` into integral bins,
    depending on each feature's *feature property*. Features
    with discrete preprocessing (not continuous, but ordered or unordered) are
    enumerated by their unique values, ascending from the lowest (Thus, a
    column with ``10, 11, 12`` would be binned as ``0, 1, 2``).

    If no ``feature_properties`` are passed, all columns in ``X`` are treated
    as :obj:`cyclic_boosting.flags.IS_CONTINUOUS`. If a ``feature_properties``
    dictionary is supplied, it must contain feature properties for each feature
    in ``X``.

    Not-a-number values in the input feature matrix are mapped to
    :obj:`cyclic_boosting.binning.MISSING_VALUE_AS_BINNO` in the transform
    step. This value can then be treated as a missing value by Cyclic Boosting.

    The feature property :obj:`cyclic_boosting.flags.HAS_MAGIC_INT_MISSING`
    enables missing-value treatment for values of -999 and -9 in integer-typed
    feature columns (for both continuous and non-continuous features).

    Binning is performed for each feature-column individually. For example, two
    columns with the same value range can end up with totally different bin
    numbers. Also, the ``n_bins`` argument which is typically an integer, can
    be indivualized by passing a dict that provides column-names and the
    respective number of bins, that should be used for continuous
    preprocessing.

    During the fit, all features are treated in the same way as in
    :class:`ECdfTransformer`. During the transform step, each feature value is
    transformed to the number of its feature bin.  The range of bin numbers
    is::

      [0, trans.bins_and_cdfs_[feature_no][1].shape[0] - 1)

    For the **estimated parameters** see :class:`ECdfTransformer`.

    Parameters
    ----------
    n_bins: int
        Maximum number of bins used to estimate the empirical CDF. ``n_bins`` is
        ignored for features with discrete preprocessing.
        If a dict is passed, the feature names/indices should be the keys and the
        n_bins are the values. Example : ``{'feature a': 150, 'feature b': 20}``

    feature_properties: dict
        Dictionary listing the names of all features as keys and their
        preprocessing flags as values. When using a numpy feature matrix X with
        no column names the keys of the feature properties are the column
        indices.

    weight_column: str or int
        Optional column label or column index for the weight column. If not set
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

    Examples
    --------

    >>> feature_1 = np.asarray([2.1, 2.2, 2.5, 3.1, 3.3, 3.7, 4.1, 4.4])
    >>> X = np.c_[feature_1]

    >>> from skpro.libs.cyclic_boosting.binning import BinNumberTransformer
    >>> trans = BinNumberTransformer(n_bins=4, epsilon=1e-8)
    >>> trans = trans.fit(X)

    >>> # only one input column
    >>> column, epsilon, bins_cdfs = trans.bins_and_cdfs_[0]
    >>> assert column == 0, np.allclose(epsilon, 1e-8 * 0.1)
    >>> bins_cdfs
    array([[ 2.1 ,  0.  ],
           [ 2.2 ,  0.25],
           [ 3.1 ,  0.5 ],
           [ 3.7 ,  0.75],
           [ 4.4 ,  1.  ]])

    >>> X_test = np.c_[[1.9, 2.15, 2.4, 2.2, 3.6, 3.5, 4.3, 5.1]]
    >>> trans.transform(X_test)
    array([[0],
           [0],
           [1],
           [0],
           [2],
           [2],
           [3],
           [3]], dtype=int8)
    """

    def __init__(
        self,
        n_bins=100,
        feature_properties=None,
        weight_column=None,
        epsilon=1e-9,
        tolerance=0.1,
        inplace=False,
    ):
        self.n_bins = n_bins
        self.feature_properties = feature_properties
        self.weight_column = weight_column
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.nan_representation = MISSING_VALUE_AS_BINNO
        self.inplace = inplace
        ECdfTransformer.__init__(
            self,
            n_bins=self.n_bins,
            feature_properties=self.feature_properties,
            weight_column=self.weight_column,
            epsilon=self.epsilon,
            tolerance=self.tolerance,
        )

    def _transform_one_feature(self, X, feature_prop, col, epsilon, bins_and_cdfs):
        xt = column_selector(X, col).astype(np.float64)

        def is_finite(x):
            if flags.has_magic_missing_set(feature_prop):
                return (x != -9) & (x != -999) & np.isfinite(xt)
            else:
                return np.isfinite(xt)

        if bins_and_cdfs is not None:
            finite_mask = is_finite(xt)
            xt_f = xt[finite_mask]

            if flags.is_continuous_set(feature_prop):
                ge_multi(bins_and_cdfs[1:, 0], xt_f - epsilon, 1, xt_f)
            else:
                eq_multi(
                    bins_and_cdfs[1:, 0],
                    xt_f,
                    np.arange(len(bins_and_cdfs[1:, 0]), dtype=np.float64),
                    epsilon,
                    xt_f,
                )

            xt[finite_mask] = xt_f
            # re_check for nans, which may have been brought in by the
            # binary search (values out of bounds)
            xt[~is_finite(xt)] = MISSING_VALUE_AS_BINNO
        else:
            xt = MISSING_VALUE_AS_BINNO
        return xt

    def transform(
        self, X_orig: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        self._check_input_for_transform(X_orig)

        if not self.inplace:
            X = X_orig.copy()
        else:
            X = X_orig

        if check_frame_empty(X):
            if isinstance(X, pd.DataFrame):
                X = X.astype({col: np.int8 for col, _, _ in self.bins_and_cdfs_})
            else:
                return _as_int_array_of_minimum_dtype(X)

            return X

        n_transformed_features = len(self.bins_and_cdfs_)

        for col, epsilon, bins_and_cdfs in self.bins_and_cdfs_:
            feature_prop = _read_feature_property(col, self.feature_properties)

            if feature_prop is None:
                pass
            else:
                xt = self._transform_one_feature(
                    X, feature_prop, col, epsilon, bins_and_cdfs
                )
                column_setter(X, col, xt)

        if not isinstance(X, pd.DataFrame) and n_transformed_features == X.shape[1]:
            X = _as_int_array_of_minimum_dtype(X)
        return X

    def get_feature_bin_boundaries(self):
        return {feature: probas for feature, epsilon, probas in self.bins_and_cdfs_}


def column_selector(X, column):
    """Dispatches to column selection via pandas or numpy, depending on the type of X"""
    if isinstance(X, pd.DataFrame):
        return X[column].values
    else:
        return X[:, int(column)]


def _as_int_array_of_minimum_dtype(arr):
    if isinstance(arr, int):
        maximum = abs(arr)
    elif len(arr) == 0:
        maximum = 0
    else:
        maximum = max(arr.max(), abs(arr.min()))
    if maximum <= np.iinfo(np.int8).max:
        return np.asarray(arr, dtype=np.int8)
    elif maximum <= np.iinfo(np.int16).max:
        return np.asarray(arr, dtype=np.int16)
    elif maximum <= np.iinfo(np.int32).max:
        return np.asarray(arr, dtype=np.int32)
    else:
        return np.asarray(arr, dtype=np.int64)


def column_setter(X, column, rhs):
    """Dispatches to column selection via pandas or numpy, depending on the type of X"""
    if isinstance(X, pd.DataFrame):
        X[column] = _as_int_array_of_minimum_dtype(rhs)
    else:
        X[:, int(column)] = rhs
