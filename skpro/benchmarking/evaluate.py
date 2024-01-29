# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Utility for benchmark evaluation of probabilistic regression models."""
# based on the sktime utility of the same name

__author__ = ["fkiraly", "hazrulakmal"]
__all__ = ["evaluate"]

import time
import warnings

import numpy as np
import pandas as pd

from skpro.utils.parallel import parallelize
from skpro.utils.validation._dependencies import _check_soft_dependencies


def _split(X, y, C, train, test):
    results = dict()
    # split data according to cv
    results["X_train"] = X.iloc[train]
    results["X_test"] = X.iloc[test]

    if y is not None:
        results["y_train"] = y.iloc[train]
        results["y_test"] = y.iloc[test]

    if C is not None:
        results["C_train"] = C.iloc[train]
        results["C_test"] = C.iloc[test]

    return results


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
    backend_params=None,
    C=None,
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

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
          but changes the return to (lazy) ``dask.dataframe.DataFrame``.

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    compute : bool, default=True
        If backend="dask", whether returned DataFrame is computed.
        If set to True, returns `pd.DataFrame`, otherwise `dask.dataframe.DataFrame`.

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed,
          e.g., ``scheduler``

    C : pd.DataFrame, optional (default=None)
        censoring information to use in the evaluation experiment,
        should have same column name as y, same length as X and y
        should have entries 0 and 1 (float or int)
        0 = uncensored, 1 = (right) censored
        if None, all observations are assumed to be uncensored
        Can be passed to any probabilistic regressor,
        but is ignored if capability:survival tag is False.

    Returns
    -------
    results : pd.DataFrame or dask.dataframe.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the estimator.
        Row index is splitter index of train/test fold in ``cv``.
        Entries in the i-th row are for the i-th train/test split in ``cv``.
        Columns are as follows:
        - test_{scoring.name}: (float) Model performance score.
          If ``scoring`` is a list,
          then there is a column withname ``test_{scoring.name}`` for each scorer.
        - fit_time: (float) Time in sec for ``fit`` on train fold.
        - pred_time: (float) Time in sec to ``predict`` from fitted estimator.
        - pred_[method]_time: (float)
          Time in sec to run ``predict_[method]`` from fitted estimator.
        - len_y_train: (int) length of y_train.
        - y_train: (pd.Series) only present if see ``return_data=True``
          train fold of the i-th split in ``cv``, used to fit the estimator.
        - y_pred: (pd.Series) present if see ``return_data=True``
          predictions from fitted estimator for the i-th test fold indices of ``cv``.
        - y_test: (pd.Series) present if see ``return_data=True``
          testing fold of the i-th split in ``cv``, used to compute the metric.

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

    scoring = _check_scores(scoring)

    _evaluate_fold_kwargs = {
        "estimator": estimator,
        "scoring": scoring,
        "return_data": return_data,
        "error_score": error_score,
    }

    def gen_X_y_train_test(X, y, C, cv):
        """Generate joint splits of X, y as per cv.

        Yields
        ------
        X_train : i-th train split of y as per cv.
        X_test : i-th test split of y as per cv.
        y_train : i-th train split of y as per cv. None if y was None.
        y_test : i-th test split of y as per cv. None if y was None.
        C_train : i-th train split of C as per cv. None if C was None.
        C_test : i-th test split of C as per cv. None if C was None.
        """
        for train, test in cv.split(X):
            yield _split(X, y, C, train, test)

    # generator for X and y splits to iterate over below
    xy_splits = gen_X_y_train_test(X, y, C, cv)

    if backend == "dask":
        backend_in = "dask_lazy"
    else:
        backend_in = backend

    # dispatch by backend
    results = parallelize(
        fun=_evaluate_fold,
        iter=xy_splits,
        meta=_evaluate_fold_kwargs,
        backend=backend_in,
        backend_params=backend_params,
    )

    # final formatting of dask dataframes
    if backend in ["dask", "dask_lazy"]:
        import dask.dataframe as dd

        metadata = _get_column_order_and_datatype(scoring, return_data)

        results = dd.from_delayed(results, meta=metadata)
        if backend == "dask":
            results = results.compute()
    else:
        results = pd.concat(results)

    # final formatting of results DataFrame
    results = results.reset_index(drop=True)

    return results


def _evaluate_fold(x, meta):
    # unpack args
    X_train = x["X_train"]
    X_test = x["X_test"]
    y_train = x["y_train"]
    y_test = x["y_test"]
    C_train = x.get("C_train", None)
    C_test = x.get("C_test", None)

    estimator = meta["estimator"]
    scoring = meta["scoring"]
    return_data = meta["return_data"]
    error_score = meta["error_score"]

    # set default result values in case estimator fitting fails
    score = error_score
    fit_time = np.nan
    pred_time = np.nan
    y_pred = pd.NA

    # results and cache dictionaries
    temp_result = dict()
    y_preds_cache = dict()

    try:
        # fit/update
        start_fit = time.perf_counter()

        estimator = estimator.clone()
        estimator.fit(X_train, y_train, C=C_train)

        fit_time = time.perf_counter() - start_fit

        pred_type = {
            "pred_quantiles": "predict_quantiles",
            "pred_interval": "predict_interval",
            "pred_proba": "predict_proba",
            None: "predict",
        }
        # predict
        start_pred = time.perf_counter()
        # cache prediction from the first scitype and reuse it to compute other metrics
        for scitype in scoring:
            method = getattr(estimator, pred_type[scitype])
            for metric in scoring.get(scitype):
                pred_args = _get_pred_args_from_metric(scitype, metric)
                if pred_args == {}:
                    time_key = f"{scitype}_time"
                    result_key = f"test_{metric.name}"
                    y_pred_key = f"y_{scitype}"
                else:
                    argval = list(pred_args.values())[0]
                    time_key = f"{scitype}_{argval}_time"
                    result_key = f"test_{metric.name}_{argval}"
                    y_pred_key = f"y_{scitype}_{argval}"

                # make prediction
                if y_pred_key not in y_preds_cache.keys():
                    start_pred = time.perf_counter()
                    y_pred = method(X_test, **pred_args)
                    pred_time = time.perf_counter() - start_pred
                    temp_result[time_key] = [pred_time]
                    y_preds_cache[y_pred_key] = [y_pred]
                else:
                    y_pred = y_preds_cache[y_pred_key][0]

                # score prediction and store score
                score = metric(y_test, y_pred, y_train=y_train, C_true=C_test)
                temp_result[result_key] = [score]

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

    # format results data frame and return
    temp_result["fit_time"] = [fit_time]
    temp_result["pred_time"] = [pred_time]
    temp_result["len_y_train"] = [len(y_train)]
    if return_data:
        temp_result["y_train"] = [y_train]
        temp_result["y_test"] = [y_test]
        temp_result.update(y_preds_cache)
    result = pd.DataFrame(temp_result)
    result = result.astype({"len_y_train": int})

    column_order = _get_column_order_and_datatype(scoring, return_data)
    result = result.reindex(columns=column_order.keys())

    return result


def _get_pred_args_from_metric(scitype, metric):
    pred_args = {
        "pred_quantiles": "alpha",
        "pred_interval": "coverage",
    }
    if scitype in pred_args.keys():
        val = getattr(metric, pred_args[scitype], None)
        if val is not None:
            return {pred_args[scitype]: val}
    return {}


def _get_column_order_and_datatype(metric_types, return_data=True):
    """Get the ordered column name and input datatype of results."""
    others_metadata = {"len_y_train": "int"}
    y_metadata = {
        "y_train": "object",
        "y_test": "object",
    }
    fit_metadata, metrics_metadata = {"fit_time": "float"}, {}
    for scitype in metric_types:
        for metric in metric_types.get(scitype):
            pred_args = _get_pred_args_from_metric(scitype, metric)
            if pred_args == {}:
                time_key = f"{scitype}_time"
                result_key = f"test_{metric.name}"
                y_pred_key = f"y_{scitype}"
            else:
                argval = list(pred_args.values())[0]
                time_key = f"{scitype}_{argval}_time"
                result_key = f"test_{metric.name}_{argval}"
                y_pred_key = f"y_{scitype}_{argval}"
            fit_metadata[time_key] = "float"
            metrics_metadata[result_key] = "float"
            if return_data:
                y_metadata[y_pred_key] = "object"
    fit_metadata.update(others_metadata)
    if return_data:
        fit_metadata.update(y_metadata)
    metrics_metadata.update(fit_metadata)
    return metrics_metadata.copy()


def _check_scores(metrics):
    """Validate and coerce to BaseMetric and segregate them based on predict type.

    Parameters
    ----------
    metrics : sktime accepted metrics object or a list of them or None

    Return
    ------
    metrics_type : Dict
        The key is metric types and its value is a list of its corresponding metrics.
    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    metrics_type = {}
    for metric in metrics:
        # collect predict type
        if hasattr(metric, "get_tag"):
            scitype = metric.get_tag(
                "scitype:y_pred", raise_error=False, tag_value_default="pred"
            )
        else:  # If no scitype exists then metric is a point forecast type
            scitype = "pred"
        if scitype not in metrics_type.keys():
            metrics_type[scitype] = [metric]
        else:
            metrics_type[scitype].append(metric)
    return metrics_type
