import logging

import numpy as np
import pandas as pd

from skpro.libs.cyclic_boosting import flags

_logger = logging.getLogger(__name__)


def _read_feature_property(col, feature_properties=None):
    """
    Get the feature property for a specific column

    Parameters
    ----------
    col: int or str
        column index specifier, integer for :class:`numpy.ndarray` or
        :class:`str` for :class:`pandas.DataFrame`

    feature_properties: dict
        Dictionary listing the names of all features as keys and their
        preprocessing flags as values. When using a numpy feature matrix X with
        no column names the keys of the feature properties are the column
        indices.

    Returns
    -------
    int
        feature property
    """
    if feature_properties is None:
        return flags.IS_CONTINUOUS
    else:
        try:
            fprop = feature_properties[col]
            flags.check_flags_consistency(fprop)
        except KeyError:
            _logger.warning(
                "Column '%s' not found in " "feature_properties dict." % col
            )
            fprop = None
        return fprop


def minimal_difference(values):
    """Minimal difference of consecutive array values
    excluding zero differences.

    :param values: Array values
    :type values: :class:`numpy.ndarray` with dim=1.
    """
    bin_widths = values[1:] - values[:-1]
    bin_widths = bin_widths[bin_widths > 0]

    if len(bin_widths) > 0:
        return np.min(bin_widths)
    else:
        return 1


def get_column_index(X, column_name_or_index):
    """Integer column index of pandas.Dataframe or numpy.ndarray.

    :param X: input matrix
    :type X: numpy.ndarray(dim=2) or pandas.DataFrame

    :param column_name_or_index: column name or index
    :type column_name_or_index: string or int

    :rtype: int
    """
    if isinstance(X, pd.DataFrame):
        return list(X.columns).index(column_name_or_index)
    else:
        return column_name_or_index


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


def check_frame_empty(X):
    """Check if a :class:`pd.DataFrame` or a :class:`numpy.ndarray`
    is empty.

    :param X: input matrix
    :type X: :class:`pd.DataFrame` or a :class:`numpy.ndarray`
    """
    if isinstance(X, pd.DataFrame):
        return X.empty
    else:
        return X.size == 0
