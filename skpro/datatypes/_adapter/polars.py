# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common utilities for polars based data containers."""
import pandas as pd

from skpro.datatypes._common import _req
from skpro.datatypes._common import _ret as ret


def check_polars_frame(obj, return_metadata=False, var_name="obj", lazy=False):
    """Check polars frame, generic format."""
    import polars as pl

    metadata = {}

    if lazy:
        exp_type = pl.LazyFrame
        exp_type_str = "LazyFrame"
    else:
        exp_type = pl.DataFrame
        exp_type_str = "DataFrame"

    if not isinstance(obj, exp_type):
        msg = f"{var_name} must be a polars {exp_type_str}, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a polars DataFrame or LazyFrame
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = obj.width < 1
    if _req("is_univariate", return_metadata):
        obj_width = obj.width
        for col in obj.columns:
            if "__index__" in col:
                obj_width -= 1
        metadata["is_univariate"] = obj_width == 1
    if _req("n_instances", return_metadata):
        if hasattr(obj, "height"):
            metadata["n_instances"] = obj.height
        else:
            metadata["n_instances"] = "NA"
    if _req("n_features", return_metadata):
        obj_width = obj.width
        for col in obj.columns:
            if "__index__" in col:
                obj_width -= 1
        metadata["n_features"] = obj_width
    if _req("feature_names", return_metadata):
        if lazy:
            obj_columns = obj.collect_schema().names()
            feature_names = [
                col for col in obj_columns if not col.startswith("__index__")
            ]
            metadata["feature_names"] = feature_names
        else:
            obj_columns = obj.columns
            feature_names = [
                col for col in obj_columns if not col.startswith("__index__")
            ]
            metadata["feature_names"] = feature_names

    # check if there are any nans
    #   compute only if needed
    if _req("has_nans", return_metadata):
        if isinstance(obj, pl.LazyFrame):
            metadata["has_nans"] = "NA"
        else:
            hasnan = obj.null_count().sum_horizontal().to_numpy()[0] > 0
            metadata["has_nans"] = hasnan

    return ret(True, None, metadata, return_metadata)


def convert_polars_to_pandas_with_index(obj):
    """Convert function from polars to pandas,converts  __index__ to pandas index.

    Parameters
    ----------
    obj : polars DataFrame, polars.LazyFrame

    Returns
    -------
    pd_df : pandas DataFrame
        Returned is a pandas DataFrame with index retained if column __index__
        existed in the polars dataframe previously, if not then index of
        pd_df will be a RangeIndex from 0 to pd_df.shape[0]-1.

    """
    from polars.lazyframe.frame import LazyFrame

    if isinstance(obj, LazyFrame):
        obj = obj.collect()

    pd_df = obj.to_pandas()
    for col in obj.columns:
        if col.startswith("__index__"):
            pd_df = pd_df.set_index(col, drop=True)
            pd_df.index.name = col.split("__index__")[1]

    return pd_df


def convert_pandas_to_polars_with_index(
    obj, schema_overrides=None, rechunk=True, nan_to_null=True, lazy=False
):
    """Convert function from pandas to polars, and preserves index.

    Parameters
    ----------
    obj : pandas DataFrame

    schema_overrides : dict, optional (default=None)
        Support override of inferred types for one or more columns.

    rechunk : bool, optional (default=True)
        Make sure that all data is in contiguous memory.

    nan_to_null : bool, optional (default=True)
        If data contains NaN values PyArrow will convert the NaN to None

    lazy : bool, optional (default=False)
        If True, return a LazyFrame instead of a DataFrame

    Returns
    -------
    pl_df : polars DataFrame or polars LazyFrame
        index from pandas DataFrame will be returned as a polars column
        named __index__.
    """
    import pandas as pd
    from polars import from_pandas

    # if the index of the dataframe is the trivial index (i.e RangeIndex(0,numrows))
    # we do not return an __index__ column
    if not (
        isinstance(obj.index, pd.RangeIndex)
        and obj.index.start == 0
        and obj.index.stop == len(obj)
    ):
        obj_index_name = obj.index.name
        obj = obj.reset_index()
        if obj_index_name is not None:
            obj = obj.rename(columns={obj_index_name: f"__index__{obj_index_name}"})
        else:
            obj = obj.rename(columns={"index": "__index__"})

    n_column_levels = check_n_level_of_dataframe(obj)
    if n_column_levels > 1:
        obj.columns = transform_pandas_multiindex_columns_to_single_column(obj)

    pl_df = from_pandas(
        data=obj,
        schema_overrides=schema_overrides,
        rechunk=rechunk,
        nan_to_null=nan_to_null,
    )

    if lazy:
        pl_df = pl_df.lazy()

    return pl_df


def transform_pandas_multiindex_columns_to_single_column(X_input: pd.DataFrame):
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


def transform_single_column_to_muldiindex_columns(obj):
    """Convert function to return a list containing un-melted columns."""
    obj_columns = obj.columns

    col_array = []
    for col in obj_columns:
        items = col.split("__")
        items = [item for item in items if item]
        col_array.append(items)


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
    import polars as pl

    if axis not in [0, 1]:
        raise ValueError(f"axis must be in [0,1] " f"found {axis}.")
    levels = None
    if isinstance(X_input, pd.DataFrame):
        if axis == 0:
            levels = X_input.index.nlevels
        elif axis == 1:
            levels = X_input.columns.nlevels

    if isinstance(X_input, pl.DataFrame) or isinstance(X_input, pl.LazyFrame):
        levels = 1

    return levels
