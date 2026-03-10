"""Utility functions for adapting to sklearn."""

import numpy as np
import pandas as pd


def prep_skl_df(df: pd.DataFrame, copy_df: bool = False) -> pd.DataFrame:
    """Make DataFrame compatible with sklearn input expectations.

    Changes
    -------
    Ensures that the column index consists of strings by converting
    column names to string type if necessary.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to make compatible with sklearn.
    copy_df : bool, default=False
        Whether to mutate ``df`` or return a copy.
        If True, the original DataFrame is not modified.

    Returns
    -------
    pd.DataFrame
        sklearn-compatible DataFrame.
    """
    cols = df.columns
    str_cols = cols.astype(str)

    if not np.all(str_cols == cols):
        if copy_df:
            df = df.copy()
        df.columns = str_cols

    return df
