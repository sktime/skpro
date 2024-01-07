"""Testing utility to compare equality in value for nested objects.

Objects compared can have one of the following valid types:
    types compatible with != comparison
    pd.Series, pd.DataFrame, np.ndarray
    lists, tuples, or dicts of a valid type (recursive)
    polars.DataFrame, polars.LazyFrame
"""
from skbase.utils.deep_equals._common import _make_ret
from skbase.utils.deep_equals._deep_equals import deep_equals as _deep_equals

from skpro.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["fkiraly"]
__all__ = ["deep_equals"]


def deep_equals(x, y, return_msg=False, plugins=None):
    """Test two objects for equality in value.

    Correct if x/y are one of the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)

    Important note:
        this function will return "not equal" if types of x,y are different
        for instant, bool and numpy.bool are *not* considered equal

    Parameters
    ----------
    x : object
    y : object
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal
    plugins : list, optional, default=None
        optional additional deep_equals plugins to use
        will be appended to the default plugins from ``deep_equals_custom``
        see ``deep_equals_custom`` for details of signature of plugins

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
            concatenation of the following strings:
            .type - type is not equal
            .class - both objects are classes but not equal
            .len - length is not equal
            .value - value is not equal
            .keys - if dict, keys of dict are not equal
                    if class/object, names of attributes and methods are not equal
            .dtype - dtype of pandas or numpy object is not equal
            .index - index of pandas object is not equal
            .series_equals, .df_equals, .index_equals - .equals of pd returns False
            [i] - if tuple/list: i-th element not equal
            [key] - if dict: value at key is not equal
            [colname] - if pandas.DataFrame: column with name colname is not equal
            != - call to generic != returns False
            .polars_equals - .equals of polars returns False
    """
    # call deep_equals_custom with default plugins
    plugins_default = [
        _polars_equals_plugin,
    ]

    if plugins is not None:
        plugins_inner = plugins_default + plugins
    else:
        plugins_inner = plugins_default

    res = _deep_equals(x, y, return_msg=return_msg, plugins=plugins_inner)
    return res


def _polars_equals_plugin(x, y, return_msg=False):
    polars_available = _check_soft_dependencies("polars", severity="none")

    if not polars_available:
        return None

    import polars as pl

    if not isinstance(x, (pl.DataFrame, pl.LazyFrame)):
        return None

    ret = _make_ret(return_msg)

    # compare pl.DataFrame
    if isinstance(x, pl.DataFrame):
        return ret(x.equals(y), ".polars_equals")

    # compare pl.LazyFrame
    if isinstance(x, pl.LazyFrame):
        return ret(x.collect().equals(y.collect()), ".polars_equals")

    return None
