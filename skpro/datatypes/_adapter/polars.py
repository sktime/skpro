# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common utilities for polars based data containers."""
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
        metadata["is_univariate"] = obj.width == 1
    if _req("n_instances", return_metadata):
        if hasattr(obj, "height"):
            metadata["n_instances"] = obj.height
        else:
            metadata["n_instances"] = "NA"
    if _req("n_features", return_metadata):
        metadata["n_features"] = obj.width
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj.columns

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
    from polars import from_pandas

    obj_index_name = obj.index.name
    obj.reset_index()
    obj.rename(columns={obj_index_name: f"__index__{obj_index_name}"})

    pl_df = from_pandas(
        data=obj,
        schema_overrides=schema_overrides,
        rechunk=rechunk,
        nan_to_null=nan_to_null,
    )

    if lazy:
        pl_df = pl_df.lazy()

    return pl_df
