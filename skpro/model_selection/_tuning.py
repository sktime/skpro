# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tuning of probabilistic supervised regressors."""

__author__ = ["fkiraly"]
__all__ = ["GridSearchCV", "RandomizedSearchCV"]

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler, check_cv

from skpro.benchmarking.evaluate import evaluate
from skpro.regression.base._delegate import _DelegatedProbaRegressor
from skpro.utils.parallel import parallelize


class BaseGridSearch(_DelegatedProbaRegressor):
    _tags = {
        "estimator_type": "regressor",
        "capability:multioutput": True,
        "capability:missing": True,
    }

    # todo 2.3.0: remove pre_dispatch and n_jobs params
    def __init__(
        self,
        estimator,
        cv,
        n_jobs=None,
        pre_dispatch=None,
        backend="loky",
        refit=True,
        scoring=None,
        verbose=0,
        return_n_best_estimators=1,
        error_score=np.nan,
        backend_params=None,
    ):
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.backend = backend
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        self.return_n_best_estimators = return_n_best_estimators
        self.error_score = error_score
        self.backend_params = backend_params

        super().__init__()

        tags_to_clone = [
            "capability:multioutput",
            "capability:missing",
            "capability:survival",
        ]
        self.clone_tags(estimator, tags_to_clone)

    # attribute for _DelegatedProbaRegressor, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedProbaRegressor docstring
    _delegate_name = "best_estimator_"

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
            A dict containing the best hyper parameters and the parameters of
            the best estimator (if available), merged together with the former
            taking precedence.
        """
        fitted_params = {}
        try:
            fitted_params = self.best_estimator_.get_fitted_params()
        except NotImplementedError:
            pass
        fitted_params = {**fitted_params, **self.best_params_}
        fitted_params.update(self._get_fitted_params_default())

        return fitted_params

    def _run_search(self, evaluate_candidates):
        raise NotImplementedError("abstract method")

    def _fit(self, X, y, C=None):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pd.DataFrame, must be same length as X
            labels to fit regressor to
        C : pd.DataFrame, optional (default=None)
            censoring information for survival analysis,
            should have same column name as y, same length as X and y
            should have entries 0 and 1 (float or int)
            0 = uncensored, 1 = (right) censored
            if None, all observations are assumed to be uncensored
            Can be passed to any probabilistic regressor,
            but is ignored if capability:survival tag is False.

        Returns
        -------
        self : reference to self
        """
        cv = check_cv(self.cv)

        # scoring = check_scoring(self.scoring, obj=self)
        scoring = self.scoring
        scoring_name = f"test_{scoring.name}"

        # todo 2.3.0: remove this logic and only use backend_params
        backend = self.backend
        backend_params = self.backend_params if self.backend_params else {}
        if backend in ["threading", "multiprocessing", "loky"]:
            n_jobs = self.n_jobs
            pre_dispatch = self.pre_dispatch
            backend_params["n_jobs"] = n_jobs
            backend_params["pre_dispatch"] = pre_dispatch
            if n_jobs is not None or pre_dispatch is not None:
                warn(
                    f"in {self.__class__.__name__}, n_jobs and pre_dispatch "
                    "parameters are deprecated and will be removed in 2.3.0. "
                    "Please use n_jobs and pre_dispatch directly in the backend_params "
                    "argument instead.",
                    stacklevel=2,
                )

        def _fit_and_score(params, meta):
            # Clone estimator.
            estimator = self.estimator.clone()

            # Set parameters.
            estimator.set_params(**params)

            # Evaluate.
            out = evaluate(
                estimator,
                cv=cv,
                X=X,
                y=y,
                C=C,
                scoring=scoring,
                error_score=self.error_score,
            )

            # Filter columns.
            out = out.filter(items=[scoring_name, "fit_time", "pred_time"], axis=1)

            # Aggregate results.
            out = out.mean()
            out = out.add_prefix("mean_")

            # Add parameters to output table.
            out["params"] = params

            return out

        def evaluate_candidates(candidate_params):
            candidate_params = list(candidate_params)

            if self.verbose > 0:
                n_candidates = len(candidate_params)
                n_splits = cv.get_n_splits(y)
                print(  # noqa
                    "Fitting {} folds for each of {} candidates,"
                    " totalling {} fits".format(
                        n_splits, n_candidates, n_candidates * n_splits
                    )
                )

            out = parallelize(
                fun=_fit_and_score,
                iter=candidate_params,
                backend=backend,
                backend_params=backend_params,
            )

            if len(out) < 1:
                raise ValueError(
                    "No fits were performed. "
                    "Was the CV iterator empty? "
                    "Were there no candidates?"
                )

            return out

        # Run grid-search cross-validation.
        results = self._run_search(evaluate_candidates)

        results = pd.DataFrame(results)

        # Rank results, according to whether greater is better for the given scoring.
        results[f"rank_{scoring_name}"] = results.loc[:, f"mean_{scoring_name}"].rank(
            ascending=scoring.get_tag("lower_is_better")
        )

        self.cv_results_ = results

        # Select best parameters.
        self.best_index_ = results.loc[:, f"rank_{scoring_name}"].argmin()
        # Raise error if all fits in evaluate failed because all score values are NaN.
        if self.best_index_ == -1:
            raise RuntimeError(
                f"""All fits of estimator failed,
                set error_score='raise' to see the exceptions.
                Failed estimator: {self.estimator}"""
            )
        self.best_score_ = results.loc[self.best_index_, f"mean_{scoring_name}"]
        self.best_params_ = results.loc[self.best_index_, "params"]
        self.best_estimator_ = self.estimator.clone().set_params(**self.best_params_)

        # Refit model with best parameters.
        if self.refit:
            self.best_estimator_.fit(X, y, C=C)

        # Sort values according to rank
        results = results.sort_values(
            by=f"rank_{scoring_name}",
            ascending=True,
        )
        # Select n best estimator
        self.n_best_estimators_ = []
        self.n_best_scores_ = []
        for i in range(self.return_n_best_estimators):
            params = results["params"].iloc[i]
            rank = results[f"rank_{scoring_name}"].iloc[i]
            rank = str(int(rank))
            estimator = self.estimator.clone().set_params(**params)
            # Refit model with best parameters.
            if self.refit:
                estimator.fit(X, y)
            self.n_best_estimators_.append((rank, estimator))
            # Save score
            score = results[f"mean_{scoring_name}"].iloc[i]
            self.n_best_scores_.append(score)

        return self

    def _get_delegate(self):
        if self.is_fitted and not self.refit:
            raise RuntimeError(
                f"In {self.__class__.__name__}, refit must be True to make predictions,"
                f" but found refit=False. If refit=False, {self.__class__.__name__} can"
                " be used only to tune hyper-parameters, as a parameter estimator."
            )
        return getattr(self, self._delegate_name)


class GridSearchCV(BaseGridSearch):
    """Perform grid-search cross-validation to find optimal model parameters.

    The estimator is fit on the initial window and then temporal
    cross-validation is used to find the optimal parameter.

    Grid-search cross-validation is performed based on a cross-validation
    iterator encoding the cross-validation scheme, the parameter grid to
    search over, and (optionally) the evaluation metric for comparing model
    performance. As in scikit-learn, tuning works through the common
    hyper-parameter interface which allows to repeatedly fit and evaluate
    the same estimator with different hyper-parameters.

    Parameters
    ----------
    estimator : estimator object
        The estimator should implement the skpro estimator
        interface. Either the estimator must contain a "score" function,
        or a scoring function must be passed.

    cv : cross-validation generator or an iterable
        e.g. KFold(n_splits=3)

    param_grid : dict or list of dictionaries
        Model tuning parameters of the estimator to evaluate

    scoring : skpro metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the estimator

        * skpro metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        `(y_true: pd.DataFrame, y_pred: BaseDistribution) -> float`,
        assuming y_true, y_pred are of the same length, lower being better,
        Metrics in skpro.metrics are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to CRPS()

    n_jobs: int, optional (default=None)
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    refit : bool, optional (default=True)
        True = refit the estimator with the best parameters on the entire data in fit
        False = no refitting takes place. The estimator cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsestimator.
    verbose: int, optional (default=0)
    return_n_best_estimators : int, default=1
        In case the n best estimator should be returned, this value can be set
        and the n best estimators will be assigned to n_best_estimators_
    pre_dispatch : str, optional (default='2*n_jobs')
    error_score : numeric value or the str 'raise', optional (default=np.nan)
        The test score returned when a estimator fails to be fitted.
    return_train_score : bool, optional (default=False)

    backend : {"dask", "loky", "multiprocessing", "threading"}, by default "loky".
        Runs parallel evaluate if specified and `strategy` is set as "refit".

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

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
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g., ``scheduler``

    Attributes
    ----------
    best_index_ : int
    best_score_: float
        Score of the best model
    best_params_ : dict
        Best parameter values across the parameter grid
    best_estimator_ : estimator
        Fitted estimator with the best parameters
    cv_results_ : dict
        Results from grid search cross validation
    n_splits_: int
        Number of splits in the data for cross validation
    refit_time_ : float
        Time (seconds) to refit the best estimator
    scorer_ : function
        Function used to score model
    n_best_estimators_: list of tuples ("rank", <estimator>)
        The "rank" is in relation to best_estimator_
    n_best_scores_: list of float
        The scores of n_best_estimators_ sorted from best to worst
        score of estimators

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import KFold, ShuffleSplit, train_test_split

    >>> from skpro.metrics import CRPS
    >>> from skpro.model_selection import GridSearchCV
    >>> from skpro.regression.residual import ResidualDouble

    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> y = pd.DataFrame(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)

    >>> cv = KFold(n_splits=3)

    >>> estimator = ResidualDouble(LinearRegression())
    >>> param_grid = {"estimator__fit_intercept" : [True, False]}
    >>> gscv = GridSearchCV(
    ...     estimator=estimator,
    ...     param_grid=param_grid,
    ...     cv=cv,
    ...     scoring=CRPS(),
    ... )
    >>> gscv.fit(X_train, y_train)
    GridSearchCV(...)
    >>> y_pred = gscv.predict(X_test)
    >>> y_pred_proba = gscv.predict_proba(X_test)
    """

    def __init__(
        self,
        estimator,
        cv,
        param_grid,
        scoring=None,
        n_jobs=None,
        refit=True,
        verbose=0,
        return_n_best_estimators=1,
        pre_dispatch="2*n_jobs",
        backend="loky",
        error_score=np.nan,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            return_n_best_estimators=return_n_best_estimators,
            pre_dispatch=pre_dispatch,
            backend=backend,
            error_score=error_score,
        )
        self.param_grid = param_grid

    def _check_param_grid(self, param_grid):
        """_check_param_grid from sklearn 1.0.2, before it was removed."""
        if hasattr(param_grid, "items"):
            param_grid = [param_grid]

        for p in param_grid:
            for name, v in p.items():
                if isinstance(v, np.ndarray) and v.ndim > 1:
                    raise ValueError("Parameter array should be one-dimensional.")

                if isinstance(v, str) or not isinstance(v, (np.ndarray, list)):
                    raise ValueError(
                        "Parameter grid for parameter ({}) needs to"
                        " be a list or numpy array, but got ({})."
                        " Single values need to be wrapped in a list"
                        " with one element.".format(name, type(v))
                    )

                if len(v) == 0:
                    raise ValueError(
                        "Parameter values for parameter ({}) need "
                        "to be a non-empty sequence.".format(name)
                    )

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid."""
        self._check_param_grid(self.param_grid)
        return evaluate_candidates(ParameterGrid(self.param_grid))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold

        from skpro.metrics import CRPS, PinballLoss
        from skpro.regression.residual import ResidualDouble
        from skpro.survival.coxph import CoxPH
        from skpro.utils.validation._dependencies import _check_estimator_deps

        linreg1 = LinearRegression()
        linreg2 = LinearRegression(fit_intercept=False)

        param1 = {
            "estimator": ResidualDouble(LinearRegression()),
            "cv": KFold(n_splits=3),
            "param_grid": {"estimator": [linreg1, linreg2]},
            "scoring": CRPS(),
            "error_score": "raise",
        }

        param2 = {
            "estimator": ResidualDouble(LinearRegression()),
            "cv": KFold(n_splits=4),
            "param_grid": {"estimator__fit_intercept": [True, False]},
            "scoring": PinballLoss(),
            "error_score": "raise",
        }

        params = [param1, param2]

        # testing with survival predictor
        if _check_estimator_deps(CoxPH, severity="none"):
            param3 = {
                "estimator": CoxPH(alpha=0.05),
                "cv": KFold(n_splits=4),
                "param_grid": {"method": ["lpl", "elastic_net"]},
                "scoring": PinballLoss(),
                "error_score": "raise",
            }
            params.append(param3)

        return params


class RandomizedSearchCV(BaseGridSearch):
    """Perform randomized-search cross-validation to find optimal model parameters.

    The estimator is fit on the initial window and then
    cross-validation is used to find the optimal parameter

    Randomized cross-validation is performed based on a cross-validation
    iterator encoding the cross-validation scheme, the parameter distributions to
    search over, and (optionally) the evaluation metric for comparing model
    performance. As in scikit-learn, tuning works through the common
    hyper-parameter interface which allows to repeatedly fit and evaluate
    the same estimator with different hyper-parameters.

    Parameters
    ----------
    estimator : estimator object
        The estimator should implement the skpro or scikit-learn estimator
        interface. Either the estimator must contain a "score" function,
        or a scoring function must be passed.

    cv : cross-validation generator or an iterable
        e.g. KFold(n_splits=3)

    param_distributions : dict or list of dicts
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : skpro metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the estimator

        * skpro metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        `(y_true: pd.DataFrame, y_pred: BaseDistribution) -> float`,
        assuming y_true, y_pred are of the same length, lower being better,
        Metrics in skpro.metrics are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to CRPS()

    n_jobs : int, optional (default=None)
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    refit : bool, optional (default=True)
        True = refit the estimator with the best parameters on the entire data in fit
        False = no refitting takes place. The estimator cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsestimator.
    verbose : int, optional (default=0)
    return_n_best_estimators: int, default=1
        In case the n best estimator should be returned, this value can be set
        and the n best estimators will be assigned to n_best_estimators_
    pre_dispatch : str, optional (default='2*n_jobs')
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
    pre_dispatch : str, optional (default='2*n_jobs')

    backend : {"dask", "loky", "multiprocessing", "threading"}, by default "loky".
        Runs parallel evaluate if specified and `strategy` is set as "refit".

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

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
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g., ``scheduler``

    Attributes
    ----------
    best_index_ : int
    best_score_: float
        Score of the best model
    best_params_ : dict
        Best parameter values across the parameter grid
    best_estimator_ : estimator
        Fitted estimator with the best parameters
    cv_results_ : dict
        Results from grid search cross validation
    n_best_estimators_: list of tuples ("rank", <estimator>)
        The "rank" is in relation to best_estimator_
    n_best_scores_: list of float
        The scores of n_best_estimators_ sorted from best to worst
        score of estimators

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import KFold, ShuffleSplit, train_test_split

    >>> from skpro.metrics import CRPS
    >>> from skpro.model_selection import RandomizedSearchCV
    >>> from skpro.regression.residual import ResidualDouble

    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> y = pd.DataFrame(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)

    >>> cv = KFold(n_splits=3)

    >>> estimator = ResidualDouble(LinearRegression())
    >>> param_distributions = {"estimator__fit_intercept" : [True, False]}
    >>> rscv = RandomizedSearchCV(
    ...     estimator=estimator,
    ...     param_distributions=param_distributions,
    ...     cv=cv,
    ...     scoring=CRPS(),
    ... )
    >>> rscv.fit(X_train, y_train)
    RandomizedSearchCV(...)
    >>> y_pred = rscv.predict(X_test)
    >>> y_pred_proba = rscv.predict_proba(X_test)
    """

    def __init__(
        self,
        estimator,
        cv,
        param_distributions,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        verbose=0,
        return_n_best_estimators=1,
        random_state=None,
        pre_dispatch="2*n_jobs",
        backend="loky",
        error_score=np.nan,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            return_n_best_estimators=return_n_best_estimators,
            pre_dispatch=pre_dispatch,
            backend=backend,
            error_score=error_score,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions."""
        return evaluate_candidates(
            ParameterSampler(
                self.param_distributions, self.n_iter, random_state=self.random_state
            )
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold

        from skpro.metrics import CRPS, PinballLoss
        from skpro.regression.residual import ResidualDouble
        from skpro.survival.coxph import CoxPH
        from skpro.utils.validation._dependencies import _check_estimator_deps

        linreg1 = LinearRegression()
        linreg2 = LinearRegression(fit_intercept=False)

        param1 = {
            "estimator": ResidualDouble(LinearRegression()),
            "cv": KFold(n_splits=3),
            "param_distributions": {"estimator": [linreg1, linreg2]},
            "scoring": CRPS(),
            "error_score": "raise",
        }

        param2 = {
            "estimator": ResidualDouble(LinearRegression()),
            "cv": KFold(n_splits=4),
            "param_distributions": {"estimator__fit_intercept": [True, False]},
            "scoring": PinballLoss(),
            "error_score": "raise",
        }

        params = [param1, param2]

        # testing with survival predictor
        if _check_estimator_deps(CoxPH, severity="none"):
            param3 = {
                "estimator": CoxPH(alpha=0.05),
                "cv": KFold(n_splits=4),
                "param_distributions": {"method": ["lpl", "elastic_net"]},
                "scoring": PinballLoss(),
                "error_score": "raise",
            }
            params += [param3]

        return params
