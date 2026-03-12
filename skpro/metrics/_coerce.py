"""Output coercion utilities for metric classes."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

import pandas as pd


def _coerce_to_scalar(obj):
    """Coerce obj to scalar, from polymorphic input scalar or pandas."""
    if isinstance(obj, pd.DataFrame):
        if len(obj) != 1:
            raise ValueError(f"Expected DataFrame with 1 row, got {len(obj)}.")
        if len(obj.columns) != 1:
            raise ValueError(
                f"Expected DataFrame with 1 column, got {len(obj.columns)}."
            )
        return obj.iloc[0, 0]
    if isinstance(obj, pd.Series):
        if len(obj) != 1:
            raise ValueError(f"Expected Series with 1 element, got {len(obj)}.")
        return obj.iloc[0]
    return obj


def _coerce_to_df(obj):
    """Coerce to pd.DataFrame, from polymorphic input scalar or pandas."""
    return pd.DataFrame(obj)


def _coerce_to_series(obj):
    """Coerce to pd.Series, from polymorphic input scalar or pandas."""
    if isinstance(obj, pd.DataFrame):
        if len(obj.columns) != 1:
            raise ValueError(
                f"Expected DataFrame with 1 column, got {len(obj.columns)}."
            )
        return obj.iloc[:, 0]
    elif isinstance(obj, pd.Series):
        return obj
    else:
        return pd.Series(obj)


def _coerce_to_1d_numpy(obj):
    """Coerce to 1D np.ndarray, from pd.DataFrame or pd.Series."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj = obj.values
    return obj.flatten()