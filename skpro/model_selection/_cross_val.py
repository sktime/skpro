# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Cross-validation utilities for probabilistic supervised regressors."""

__author__ = ["Ahmed"]
__all__ = ["cross_val_score"]

import numpy as np
import pandas as pd
from sklearn.model_selection import check_cv

from skpro.benchmarking.evaluate import evaluate


def cross_val_score(
    estimator,
    X,
    y,
    scoring=None,
    cv=5,
    error_score=np.nan,
    backend=None,
    backend_params=None,
    C=None,
):
    """Evaluate a probabilistic regressor by cross-validation.

    Returns an array of scores, one for each cross-validation fold.

    This is the skpro equivalent of :func:`sklearn.model_selection.cross_val_score`,
    adapted for probabilistic supervised regression with distributional metrics
    such as CRPS, LogLoss, PinballLoss, etc.

    Parameters
    ----------
    estimator : skpro BaseProbaRegressor descendant (concrete estimator)
        The probabilistic regressor to evaluate.

    X : pandas DataFrame
        Feature instances to use in the cross-validation experiment.

    y : pd.DataFrame, must be same length as X
        Labels to use in the cross-validation experiment.

    scoring : subclass of skpro.metrics.BaseProbaMetric or list of same,
        default=None.
        Probabilistic metric(s) to evaluate.

        * If a single metric, the returned array has one score per fold.
        * If a list of metrics, a ``pd.DataFrame`` is returned with one column
          per metric and one row per fold.
        * If None, defaults to ``CRPS()``.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.

        * None, to use the default 5-fold cross-validation
          (``sklearn.model_selection.KFold(n_splits=5)``),
        * int, to specify the number of folds in a ``KFold``,
        * An sklearn CV splitter instance, e.g. ``KFold(n_splits=3)``.

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting.
        If set to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

    backend : {"dask", "loky", "multiprocessing", "threading"}, by default None.
        Runs parallel evaluate if specified.

        - "None": executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment

    backend_params : dict, optional
        Additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by
          ``backend``.
        - "dask": any valid keys for ``dask.compute`` can be passed,
          e.g., ``scheduler``

    C : pd.DataFrame, optional (default=None)
        Censoring information for survival analysis.

        * should have same column name as y, same length as X and y
        * should have entries 0 and 1 (float or int),
          0 = uncensored, 1 = (right) censored

        If None, all observations are assumed to be uncensored.
        Can be passed to any probabilistic regressor,
        but is ignored if ``capability:survival`` tag is ``False``.

    Returns
    -------
    scores : np.ndarray of float, shape ``(n_splits,)``
        Array of scores of the estimator for each run of the cross-validation.
        Returned when ``scoring`` is a single metric (or None).

    scores : pd.DataFrame
        DataFrame with one column per metric and one row per fold.
        Returned when ``scoring`` is a list of metrics.

    See Also
    --------
    skpro.benchmarking.evaluate.evaluate :
        Evaluate estimator using re-sample folds with full result details.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import KFold
    >>>
    >>> from skpro.metrics import CRPS
    >>> from skpro.model_selection import cross_val_score
    >>> from skpro.regression.residual import ResidualDouble
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> y = pd.DataFrame(y)
    >>>
    >>> estimator = ResidualDouble(LinearRegression())
    >>> cv = KFold(n_splits=3)
    >>> crps = CRPS()
    >>>
    >>> scores = cross_val_score(estimator, X, y, scoring=crps, cv=cv)
    """
    cv = check_cv(cv)

    # Handle default scoring
    if scoring is None:
        from skpro.metrics import CRPS

        scoring = CRPS()

    # Determine if multi-metric mode
    is_multimetric = isinstance(scoring, list)

    # Run full evaluation using the evaluate utility
    results = evaluate(
        estimator=estimator,
        cv=cv,
        X=X,
        y=y,
        scoring=scoring,
        return_data=False,
        error_score=error_score,
        backend=backend,
        backend_params=backend_params,
        C=C,
    )

    # Extract scores from the results DataFrame
    score_columns = [col for col in results.columns if col.startswith("test_")]

    if is_multimetric:
        # Return a DataFrame with one column per metric
        scores = results[score_columns].copy()
        # Clean up column names by removing the "test_" prefix
        scores.columns = [col.replace("test_", "", 1) for col in scores.columns]
        return scores
    else:
        # Return a 1-D numpy array of scores for the single metric
        if len(score_columns) == 1:
            return results[score_columns[0]].to_numpy()
        else:
            # If multiple score columns from a single metric (e.g., with alpha),
            # return the first one as default behavior
            return results[score_columns[0]].to_numpy()
