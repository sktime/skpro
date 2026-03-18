"""Utility functions for adapting to sklearn."""

import numpy as np
import pandas as pd


def prep_skl_df(df: pd.DataFrame, copy_df: bool = False) -> pd.DataFrame:
    """
    Make DataFrame compatible with sklearn input expectations.

    Ensures that column names are strings, as required by sklearn.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to make compatible with sklearn.
    copy_df : bool, default=False
        Whether to mutate ``df`` or return a copy.

        If False, the column index of ``df`` may be modified in-place.
        If True, a copy of the DataFrame is created before modifying
        the column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with column names converted to string type if necessary.
    """
    cols = df.columns
    str_cols = cols.astype(str)

    if not np.all(str_cols == cols):
        if copy_df:
            df = df.copy()
        df.columns = str_cols

    return df
