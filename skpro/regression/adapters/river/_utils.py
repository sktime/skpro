"""Utilities for River adapter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["patelchaitany"]

import pandas as pd


def is_river_estimator(obj):
    """Return whether ``obj`` is a River estimator instance.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool
        True if ``obj`` is a River estimator, False otherwise.
    """
    if obj is None:
        return False

    mod = getattr(type(obj), "__module__", "")
    if not mod.startswith("river"):
        return False

    try:
        from river import base

        return isinstance(obj, base.Estimator)
    except ImportError:
        return mod.startswith("river")


def _ensure_str_columns(X):
    """Coerce DataFrame column names to strings for River compatibility."""
    if not isinstance(X, pd.DataFrame):
        return X
    if all(isinstance(c, str) for c in X.columns):
        return X
    X = X.copy()
    X.columns = X.columns.astype(str)
    return X


def _learn_batch(estimator, X, y):
    """Train a River estimator on a batch of inner skpro data.

    Parameters
    ----------
    estimator : river estimator
        Fitted or unfitted River model.
    X : pd.DataFrame
        Feature data in skpro inner mtype.
    y : pd.DataFrame
        Target data in skpro inner mtype, single column.
    """
    X = _ensure_str_columns(X)
    y_vec = y.iloc[:, 0]

    if hasattr(estimator, "learn_many"):
        estimator.learn_many(X, y_vec)
        return

    for i in range(len(X)):
        xi = X.iloc[i].to_dict()
        yi = float(y_vec.iloc[i])
        estimator.learn_one(xi, yi)


def _predict_batch(estimator, X):
    """Predict with a River estimator on inner skpro feature data.

    Parameters
    ----------
    estimator : river estimator
        Fitted River model.
    X : pd.DataFrame
        Feature data in skpro inner mtype.

    Returns
    -------
    pd.Series
        Point predictions, indexed like ``X``.
    """
    X = _ensure_str_columns(X)

    if hasattr(estimator, "predict_many"):
        preds = estimator.predict_many(X)
        if isinstance(preds, pd.Series):
            return preds
        if isinstance(preds, pd.DataFrame):
            return preds.iloc[:, 0]
        return pd.Series(preds, index=X.index)

    preds = []
    for i in range(len(X)):
        xi = X.iloc[i].to_dict()
        preds.append(estimator.predict_one(xi))
    return pd.Series(preds, index=X.index)
