# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Utility for benchmark evaluation of probabilistic regression models."""
# based on the sktime utility of the same name

__author__ = ["fkiraly"]
__all__ = ["evaluate"]

import time
import warnings

import numpy as np
import pandas as pd

from skpro.utils.validation._dependencies import _check_soft_dependencies

PANDAS_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]


def _split(X, y, train, test):
    # split data according to cv
    X_train, X_test = X.iloc[train], X.iloc[test]

    if y is None:
        y_train, y_test = None, None
    else:
        y_train, y_test = y.iloc[train], y.iloc[test]

    return X_train, X_test, y_train, y_test


def evaluate(
    estimator,
    cv,
    X,
    y,
    scoring=None,
    return_data=False,
    error_score=np.nan,
    backend=None,
    compute=True,
    **kwargs,
):
    r"""Evaluate estimator using re-sample folds.

    All-in-one statistical performance benchmarking utility for estimators
    which runs a simple backtest experiment and returns a summary pd.DataFrame.

    The experiment run is the following:

    Denote by :math:`X_{train, 1}, X_{test, 1}, \dots, X_{train, K}, X_{test, K}`
    the train/test folds produced by the generator ``cv.split(X)``
    Denote by :math:`y_{train, 1}, y_{test, 1}, \dots, y_{train, K}, y_{test, K}`
    the train/test folds produced by the generator ``cv.split(y)``.

    0. For ``i = 1`` to ``cv.get_n_folds(X)`` do:
    1. ``fit`` the ``estimator`` to :math:`X_{train, 1}`, :math:`y_{train, 1}`
    2. ``y_pred = estimator.predict``
      (or ``predict_proba`` or ``predict_quantiles``, depending on ``scoring``)
      with exogeneous data :math:`X_{test, i}`
    3. Compute ``scoring`` on ``y_pred``versus :math:`y_{test, 1}`.

    Results returned in this function's return are:
    * results of ``scoring`` calculations, from 3,  in the `i`-th loop
    * runtimes for fitting and/or predicting, from 1, 2 in the `i`-th loop
    * :math:`y_{train, i}`, :math:`y_{test, i}`, ``y_pred`` (optional)

    A distributed and-or parallel back-end can be chosen via the ``backend`` parameter.

    Parameters
    ----------
    estimator : skpro BaseProbaRegressor descendant (concrete estimator)
        skpro estimator to benchmark
    cv : sklearn splitter
        determines split of ``X`` and ``y`` into test and train folds
    X : pandas DataFrame
        Feature instances to use in evaluation experiment
    y : pd.DataFrame, must be same length as X
        Labels to used in the evaluation experiment
    scoring : subclass of skpro.performance_metrics.BaseMetric or list of same,
        default=None. Used to get a score function that takes y_pred and y_test
        arguments and accept y_train as keyword argument.
        If None, then uses scoring = MeanAbsolutePercentageError(symmetric=True).
    return_data : bool, default=False
        Returns three additional columns in the DataFrame, by default False.
        The cells of the columns contain each a pd.Series for y_train,
        y_pred, y_test.
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.
    backend : {"dask", "loky", "multiprocessing", "threading"}, by default None.
        Runs parallel evaluate if specified and `strategy` is set as "refit".
        - "loky", "multiprocessing" and "threading": uses `joblib` Parallel loops
        - "dask": uses `dask`, requires `dask` package in environment
        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (`cloudpickle`) for "dask" and "loky" is generally more robust than the
        standard `pickle` library used in "multiprocessing".
    compute : bool, default=True
        If backend="dask", whether returned DataFrame is computed.
        If set to True, returns `pd.DataFrame`, otherwise `dask.dataframe.DataFrame`.
    **kwargs : Keyword arguments
        Only relevant if backend is specified. Additional kwargs are passed into
        `dask.distributed.get_client` or `dask.distributed.Client` if backend is
        set to "dask", otherwise kwargs are passed into `joblib.Parallel`.

    Returns
    -------
    results : pd.DataFrame or dask.dataframe.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the estimator.
        Row index is splitter index of train/test fold in `cv`.
        Entries in the i-th row are for the i-th train/test split in `cv`.
        Columns are as follows:
        - test_{scoring.name}: (float) Model performance score. If `scoring` is a list,
            then there is a column withname `test_{scoring.name}` for each scorer.
        - fit_time: (float) Time in sec for `fit` or `update` on train fold.
        - pred_time: (float) Time in sec to `predict` from fitted estimator.
        - len_y_train: (int) length of y_train.
        - y_train: (pd.Series) only present if see `return_data=True`
          train fold of the i-th split in `cv`, used to fit the estimator.
        - y_pred: (pd.Series) present if see `return_data=True`
          predictions from fitted estimator for the i-th test fold indices of `cv`.
        - y_test: (pd.Series) present if see `return_data=True`
          testing fold of the i-th split in `cv`, used to compute the metric.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import KFold

    >>> from skpro.benchmarking.evaluate import evaluate
    >>> from skpro.metrics import CRPS
    >>> from skpro.regression.residual import ResidualDouble

    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> y = pd.DataFrame(y)  # skpro assumes y is pd.DataFrame

    >>> estimator = ResidualDouble(LinearRegression())
    >>> cv = KFold(n_splits=3)
    >>> crps = CRPS()

    >>> results = evaluate(estimator=estimator, X=X, y=y, cv=cv, scoring=crps)
    """
    if backend == "dask" and not _check_soft_dependencies("dask", severity="none"):
        raise RuntimeError(
            "running evaluate with backend='dask' requires the dask package installed,"
            "but dask is not present in the python environment"
        )

    # todo: input checks and coercions
    # cv = check_cv(cv, enforce_start_with_window=True)
    # if isinstance(scoring, list):
    #    scoring = [check_scoring(s) for s in scoring]
    # else:
    #     scoring = check_scoring(scoring)

    score_name = (
        f"test_{scoring.name}"
        if not isinstance(scoring, list)
        else f"test_{scoring[0].name}"
    )

    _evaluate_fold_kwargs = {
        "estimator": estimator,
        "scoring": scoring if not isinstance(scoring, list) else scoring[0],
        "return_data": True,
        "error_score": error_score,
        "score_name": score_name,
    }

    def gen_X_y_train_test(X, y, cv):
        """Generate joint splits of X, y as per cv.

        Yields
        ------
        X_train : i-th train split of y as per cv. None if X was None.
        X_test : i-th test split of y as per cv. None if X was None.
        y_train : i-th train split of y as per cv
        y_test : i-th test split of y as per cv
        """
        for train, test in cv.split(X):
            yield _split(X, y, train, test)

    # generator for X and y splits to iterate over below
    xy_splits = gen_X_y_train_test(X, y, cv)

    # dispatch by backend
    if backend is None:
        # Run temporal cross-validation sequentially
        results = []
        for X_train, X_test, y_train, y_test in xy_splits:
            result = _evaluate_fold(
                X_train,
                X_test,
                y_train,
                y_test,
                **_evaluate_fold_kwargs,
            )
            results.append(result)
        results = pd.concat(results)

    elif backend == "dask":
        # Use Dask delayed instead of joblib,
        # which uses Futures under the hood
        import dask.dataframe as dd
        from dask import delayed as dask_delayed

        results = []
        for X_train, X_test, y_train, y_test in xy_splits:
            results.append(
                dask_delayed(_evaluate_fold)(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    **_evaluate_fold_kwargs,
                )
            )
        results = dd.from_delayed(
            results,
            meta={
                score_name: "float",
                "fit_time": "float",
                "pred_time": "float",
                "len_y_train": "int",
                "y_train": "object",
                "y_test": "object",
                "y_pred": "object",
            },
        )
        if compute:
            results = results.compute()

    else:
        # Otherwise use joblib
        from joblib import Parallel, delayed

        results = Parallel(backend=backend, **kwargs)(
            delayed(_evaluate_fold)(
                X_train,
                X_test,
                y_train,
                y_test,
                **_evaluate_fold_kwargs,
            )
            for X_train, X_test, y_train, y_test in xy_splits
        )
        results = pd.concat(results)

    # final formatting of results DataFrame
    results = results.reset_index(drop=True)
    if isinstance(scoring, list):
        for s in scoring[1:]:
            results[f"test_{s.name}"] = np.nan
            for row in results.index:
                results.loc[row, f"test_{s.name}"] = s(
                    results["y_test"].loc[row],
                    results["y_pred"].loc[row],
                    y_train=results["y_train"].loc[row],
                )

    # drop pointer to data if not requested
    if not return_data:
        results = results.drop(columns=["y_train", "y_test", "y_pred"])
    results = results.astype({"len_y_train": int})

    return results


def _evaluate_fold(
    X_train,
    X_test,
    y_train,
    y_test,
    estimator,
    scoring,
    return_data,
    score_name,
    error_score,
):
    # set default result values in case estimator fitting fails
    score = error_score
    fit_time = np.nan
    pred_time = np.nan
    y_pred = pd.NA

    try:
        # fit/update
        start_fit = time.perf_counter()

        estimator = estimator.clone()
        estimator.fit(X_train, y_train)

        fit_time = time.perf_counter() - start_fit

        pred_type = {
            "pred_quantiles": "predict_quantiles",
            "pred_interval": "predict_interval",
            "pred_proba": "predict_proba",
            None: "predict",
        }
        # predict
        start_pred = time.perf_counter()

        if hasattr(scoring, "metric_args"):
            metric_args = scoring.metric_args
        else:
            metric_args = {}

        if hasattr(scoring, "get_tag"):
            scitype = scoring.get_tag("scitype:y_pred", raise_error=False)
        else:
            # If no scitype exists then metric is not proba and no args needed
            scitype = None

        methodname = pred_type[scitype]
        method = getattr(estimator, methodname)

        y_pred = method(X_test, **metric_args)

        pred_time = time.perf_counter() - start_pred

        # score
        score = scoring(y_test, y_pred, y_train=y_train)

    except Exception as e:
        if error_score == "raise":
            raise e
        else:
            warnings.warn(
                f"""
                In evaluate, fitting of estimator {type(estimator).__name__} failed,
                you can set error_score='raise' in evaluate to see
                the exception message. Fit failed for len(y_train)={len(y_train)}.
                The score will be set to {error_score}.
                Failed estimator with parameters: {estimator}.
                """,
                stacklevel=2,
            )

    result = pd.DataFrame(
        {
            score_name: [score],
            "fit_time": [fit_time],
            "pred_time": [pred_time],
            "len_y_train": [len(y_train)],
            "y_train": [y_train if return_data else pd.NA],
            "y_test": [y_test if return_data else pd.NA],
            "y_pred": [y_pred if return_data else pd.NA],
        }
    )

    return result
