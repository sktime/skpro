"""Utilities for set_output functionality."""

__author__ = ["julian-fong"]

import warnings
from copy import deepcopy

import pandas as pd

from skpro.utils.validation._dependencies import _check_soft_dependencies

SUPPORTED_OUTPUTS = ["pandas", "default"]

if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    SUPPORTED_OUTPUTS.append("polars")


def check_n_level_of_dataframe(X_input, axis=1):
    """Convert function to check the number of levels inside a pd/pl frame.

    Parameters
    ----------
    X_input : polars or pandas DataFrame
        A given polars or pandas DataFrame. Note that the polars portion of this
        code requires the soft dependencies polars and pyarrow to be installed

    axis : [0,1]
        Specify the index or columns of a pandas DataFrame. If 0, uses the index
        If 1, uses the columns. This parameter is ignored if X_input is not
        a pandas DataFrame.

    Returns
    -------
    levels : int
        An integer specifying the number of levels given a DataFrame
    """
    if axis not in [0, 1]:
        raise ValueError(f"axis must be in [0,1] " f"found {axis}.")
    levels = None
    if isinstance(X_input, pd.DataFrame):
        if axis == 0:
            levels = X_input.index.nlevels
        elif axis == 1:
            levels = X_input.columns.nlevels

    if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
        import polars as pl

        if isinstance(X_input, pl.DataFrame):
            levels = 1

    return levels


def convert_pandas_multiindex_columns_to_single_column(X_input: pd.DataFrame):
    """Convert function to return a list containing melted columns.

    Assumes a multi-index column pandas DataFrame

    Parameters
    ----------
    X : pandas DataFrame
        pandas DataFrame containing a multi-index column (nlevels > 1)

    Returns
    -------
    df_cols : a list object containing strings of all of the melted columns
    """
    df_cols = []
    for col in X_input.columns:
        df_cols.append("__" + "__".join(str(x) for x in col if x != "") + "__")
    # in case "__index__" is in one of the tuples inside X_input
    df_cols = [col.replace("____", "__") for col in df_cols]

    return df_cols


def convert_pandas_index_to_column(X_input):
    """Given a pandas DataFrame, convert the index into a single column.

    Assumes the DataFrame has a one-level index.

    Parameters
    ----------
    X_input : pandas DataFrame
        pandas DataFrame containing a single-level index

    Returns
    -------
    X_out : A copy of X with a new column containing the indices called
    "__index__"

    """
    X_input_ = deepcopy(X_input)
    X_out = X_input_.reset_index(names="__index__")

    return X_out


def convert_pandas_dataframe_to_polars_eager_with_index(X_input, include_index=False):
    """Given a pandas DataFrame, converts the Frame into a pl Frame with index.

    Assumes the input pandas DataFrame has a one-level index.

    Parameters
    ----------
    X_input : pandas DataFrame
        pandas DataFrame containing a single-level index

    include_index : bool
        Bool whether or not to include the index from the pandas DataFrame
        default = False

    Returns
    -------
    X_polars : polars eager DataFrame containing the index as a seperate column
    """
    X_input_ = deepcopy(X_input)
    X_input_.index.name = (
        "__index__" if not X_input_.index.name else X_input_.index.name
    )
    # instantiate the polars dataFrame as None in case user does not have dependencies
    X_polars = None
    if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
        import polars as pl

        if include_index:
            n_level = check_n_level_of_dataframe(X_input, axis=0)
            if n_level != 1:
                warnings.warn(
                    "pandas DataFrame does not contain a flat single level index. ",
                    " Converting the index may not have intended results.",
                    stacklevel=2,
                )
        X_polars = pl.from_pandas(X_input_, include_index=include_index)

    return X_polars


def check_transform_config(estimator):
    """Given an estimator, verify the transform key in _config is available.

    Parameters
    ----------
    estimator : a given regression estimator

    Returns
    -------
    dense_config : a dict containing keys with supported outputs.
        - "dense": specifies the data container in the transform config
            Possible values are located in SUPPORTED_OUTPUTS in
            `skpro.utils.set_output`
    """
    transform_output = estimator.get_config()["transform"]
    if transform_output not in SUPPORTED_OUTPUTS:
        raise ValueError(
            f"set_output container must be in {SUPPORTED_OUTPUTS}, "
            f"found {transform_output}."
        )
    elif transform_output != "default":
        valid = True
    else:
        valid = False
    return valid, {"dense": transform_output}
