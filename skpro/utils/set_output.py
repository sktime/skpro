"""Utilities for set_output functionality."""

__author__ = ["julian-fong"]

from copy import deepcopy

import pandas as pd

from skpro.utils.validation._dependencies import _check_soft_dependencies

SUPPORTED_OUTPUTS = ["pandas"]

if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    SUPPORTED_OUTPUTS.append("polars")


def check_column_level_of_dataframe(X_input, index_pandas=False):
    """Convert function to check the number of levels inside a pandas Frame.

    Parameters
    ----------
    X_input : polars or pandas DataFrame
        A given polars or pandas DataFrame. Note that the polars portion of this
        code requires the soft dependencies polars and pyarrow to be installed

    index_pandas : bool
        Specify the index or columns of a pandas DataFrame. If True, uses the index
        If false, uses the columns. This parameter is ignored if X_input is not
        a pandas DataFrame.

    Returns
    -------
    levels : int
        An integer specifying the number of levels given a DataFrame
    """
    if isinstance(X_input, pd.DataFrame):
        levels = None
        if index_pandas:
            levels = X_input.index.nlevels
        else:
            levels = X_input.columns.nlevels

    if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
        import polars as pl

        if isinstance(X_input, pl.DataFrame):
            levels = 1

    return levels


def convert_multiindex_columns_to_single_column(X_input: pd.DataFrame):
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
        df_cols.append("__" + "__".join(str(x) for x in col) + "__")

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


def convert_pandas_dataframe_to_polars_eager_with_index(X_input):
    """Given a pandas DataFrame, converts the Frame into a pl Frame with index.

    Assumes the input pandas DataFrame has a one-level index.

    Parameters
    ----------
    X_input : pandas DataFrame
        pandas DataFrame containing a single-level index

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

        X_polars = pl.from_pandas(X_input, include_index=True)

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
    if estimator.get_config()["transform"] not in SUPPORTED_OUTPUTS:
        raise ValueError(f"set_output container must be in {SUPPORTED_OUTPUTS}, ")

    return {"dense": estimator.get_config()["transform"]}