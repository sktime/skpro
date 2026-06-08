# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tuning of probabilistic supervised regressors."""

__author__ = ["fkiraly"]
__all__ = ["GridSearchCV", "RandomizedSearchCV"]

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler, check_cv

from skpro.metrics import CRPS
from skpro.benchmarking.evaluate import evaluate
from skpro.regression.base._delegate import _DelegatedProbaRegressor
from skpro.utils.parallel import parallelize


class BaseGridSearch(_DelegatedProbaRegressor):
    """Base class for hyperparameter search with cross-validation.

    Parameters
    ----------
    estimator : estimator object
        Probabilistic regressor to tune.
    cv : cross-validation generator or an iterable
        Cross-validation splitting strategy.
    backend : str, optional (default="loky")
        Parallel backend for candidate evaluation.
    refit : bool, optional (default=True)
        Whether to refit the best estimator on all data.
    scoring : skpro metric, str, or callable, optional (default=None)
        Metric used to score candidate parameter settings.
    verbose : int, optional (default=0)
        Verbosity level.
    return_n_best_estimators : int, optional (default=1)
        Number of top-ranked fitted estimators to retain.
    error_score : numeric or "raise", optional (default=np.nan)
        Score assigned when a candidate fit fails.
    backend_params : dict, optional
        Additional parameters passed to the parallel backend.
    update_behaviour : str, optional (default="no_update")
        one of ``{"no_update", "inner_only", "full_refit"}``

        Controls behaviour when ``update`` is called on a fitted tuner.
        See :class:`GridSearchCV` for a full description of each option.
    """

    _tags = {
        "estimator_type": "regressor",
        "capability:multioutput": True,
        "capability:missing": True,
        "capability:update": True,
    }

    def __init__(
        self,
        estimator,
        cv,
        backend="loky",
        refit=True,
        scoring=None,
        verbose=0,
        return_n_best_estimators=1,
        error_score=np.nan,
        backend_params=None,
        update_behaviour="no_update",
    ):
        self.estimator = estimator
        self.cv = cv
        self.backend = backend
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        self.return_n_best_estimators = return_n_best_estimators
        self.error_score = error_score
        self.backend_params = backend_params
        self.update_behaviour = update_behaviour

        super().__init__()

        tags_to_clone = [
            "capability:multioutput",
            "capability:missing",
            "capability:survival",
        ]
        self.clone_tags(estimator, tags_to_clone)
        self._set_update_capability_tag(estimator)

    def _set_update_capability_tag(self, estimator):
        """Set ``capability:update`` from ``update_behaviour`` and inner estimator."""
        behaviour = self.update_behaviour
        if behaviour == "no_update":
            self.set_tags(**{"capability:update": False})
        elif behaviour == "full_refit":
            self.set_tags(**{"capability:update": True})
        elif behaviour == "inner_only":
            self.clone_tags(estimator, ["capability:update"])
        else:
            raise ValueError(
                f"Unknown update_behaviour={behaviour!r}, must be one of "
                "'no_update', 'inner_only', 'full_refit'."
            )

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
        if scoring is None:
            scoring = CRPS()
        scoring_name = f"test_{scoring.name}"

        backend = self.backend
        backend_params = self.backend_params if self.backend_params else {}

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
                estimator.fit(X, y, C=C)
            self.n_best_estimators_.append((rank, estimator))
            # Save score
            score = results[f"mean_{scoring_name}"].iloc[i]
            self.n_best_scores_.append(score)

        if self.update_behaviour == "full_refit":
            self._X = X
            self._y = y
            self._C = C

        return self

    def _update(self, X, y, C=None):
        """Update fitted tuner with a new batch of training data.

        Behaviour is controlled by ``update_behaviour``:

        - ``"no_update"``: return without changing fitted state.
        - ``"inner_only"``: call ``best_estimator_.update(X, y, C=C)``.
        - ``"full_refit"``: concatenate new data with stored training data
          and re-run ``_fit`` (requires ``update_behaviour="full_refit"`` at
          construction and ``fit`` time so training data is retained).

        State required:
            Requires state to be "fitted".

        Writes to self:
            Updates fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to update regressor with
        y : pd.DataFrame, must be same length as X
            labels to update regressor with
        C : pd.DataFrame, optional (default=None)
            censoring information for survival analysis

        Returns
        -------
        self : reference to self
        """
        behaviour = self.update_behaviour

        if behaviour == "full_refit":
            if not hasattr(self, "_X"):
                raise RuntimeError(
                    f"In {self.__class__.__name__}, update_behaviour='full_refit' "
                    "requires training data to be stored at fit time. "
                    "Set update_behaviour='full_refit' before calling fit."
                )
            self._X = self._update_data(self._X, X)
            self._y = self._update_data(self._y, y)
            self._C = self._update_data(self._C, C)
            self._fit(self._X, self._y, C=self._C)
        elif behaviour == "inner_only":
            self.best_estimator_.update(X=X, y=y, C=C)
        elif behaviour == "no_update":
            pass

        return self

    def _update_data(self, X, X_new):
        """Concatenate old and new data, resetting index.

        Parameters
        ----------
        X : pandas DataFrame or None
        X_new : pandas DataFrame or None

        Returns
        -------
        X_updated : pandas DataFrame or None
            concatenated data with reset index
        """
        if X is None and X_new is None:
            return None
        if X is None:
            return X_new.reset_index(drop=True)
        if X_new is None:
            return X.reset_index(drop=True)
        return pd.concat([X, X_new], ignore_index=True)

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
        with the ``registry.all_objects`` search utility,
        for instance via ``all_objects("metric", as_dataframe=True)``

        * If callable, must have signature
        `(y_true: pd.DataFrame, y_pred: BaseDistribution) -> float`,
        assuming y_true, y_pred are of the same length, lower being better,
        Metrics in skpro.metrics are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to CRPS()

    refit : bool, optional (default=True)
        True = refit the estimator with the best parameters on the entire data in fit
        False = no refitting takes place. The estimator cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsestimator.

    verbose: int, optional (default=0)

    return_n_best_estimators : int, default=1
        In case the n best estimator should be returned, this value can be set
        and the n best estimators will be assigned to n_best_estimators_

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

    update_behaviour : str, optional (default="no_update")
        one of ``{"no_update", "inner_only", "full_refit"}``

        Behaviour of the tuner when calling ``update``.
        Only has an effect if the inner estimator supports ``update``
        (i.e., has the ``capability:update`` tag set to ``True``).

        * ``"no_update"`` = neither tuning parameters nor inner estimator
          are updated. This is the default and matches pre-update behaviour.
        * ``"inner_only"`` = tuning parameters are not re-tuned; only the
          inner estimator (``best_estimator_``) is updated via its ``update``
          method.
        * ``"full_refit"`` = accumulate all training data seen so far and
          re-run the full hyperparameter search. Training data is stored in
          memory at ``fit`` time; use only when incremental refitting is
          required, as this can be memory-intensive for large datasets.

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
        refit=True,
        verbose=0,
        return_n_best_estimators=1,
        backend="loky",
        error_score=np.nan,
        backend_params=None,
        update_behaviour="no_update",
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            refit=refit,
            cv=cv,
            verbose=verbose,
            return_n_best_estimators=return_n_best_estimators,
            backend=backend,
            error_score=error_score,
            backend_params=backend_params,
            update_behaviour=update_behaviour,
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
        from skpro.survival.compose._reduce_cond_unc import ConditionUncensored

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

        params3 = {
            "estimator": ConditionUncensored(ResidualDouble(LinearRegression())),
            "cv": KFold(n_splits=4),
            "param_grid": {"estimator__fit_intercept": [True, False]},
            "scoring": PinballLoss(),
            "error_score": "raise",
        }
        param_no_backend = {**param1, "backend": "None"}
        params = [param1, param2, params3, param_no_backend]

        from skpro.regression.online._refit import OnlineRefit

        params4 = {
            "estimator": OnlineRefit(ResidualDouble(LinearRegression())),
            "cv": KFold(n_splits=3),
            "param_grid": {"estimator__fit_intercept": [True, False]},
            "scoring": CRPS(),
            "error_score": "raise",
            "update_behaviour": "inner_only",
        }

        params = [param1, param2, params3, params4]

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
        with the ``registry.all_objects`` search utility,
        for instance via ``all_objects("metric", as_dataframe=True)``

        * If callable, must have signature
        `(y_true: pd.DataFrame, y_pred: BaseDistribution) -> float`,
        assuming y_true, y_pred are of the same length, lower being better,
        Metrics in skpro.metrics are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to CRPS()

    refit : bool, optional (default=True)
        True = refit the estimator with the best parameters on the entire data in fit
        False = no refitting takes place. The estimator cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsestimator.

    verbose : int, optional (default=0)

    return_n_best_estimators: int, default=1
        In case the n best estimator should be returned, this value can be set
        and the n best estimators will be assigned to n_best_estimators_

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.

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

    update_behaviour : str, optional (default="no_update")
        one of ``{"no_update", "inner_only", "full_refit"}``

        Behaviour of the tuner when calling ``update``.
        Only has an effect if the inner estimator supports ``update``
        (i.e., has the ``capability:update`` tag set to ``True``).

        * ``"no_update"`` = neither tuning parameters nor inner estimator
          are updated. This is the default and matches pre-update behaviour.
        * ``"inner_only"`` = tuning parameters are not re-tuned; only the
          inner estimator (``best_estimator_``) is updated via its ``update``
          method.
        * ``"full_refit"`` = accumulate all training data seen so far and
          re-run the full hyperparameter search. Training data is stored in
          memory at ``fit`` time; use only when incremental refitting is
          required, as this can be memory-intensive for large datasets.

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
        refit=True,
        verbose=0,
        return_n_best_estimators=1,
        random_state=None,
        backend="loky",
        error_score=np.nan,
        backend_params=None,
        update_behaviour="no_update",
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            refit=refit,
            cv=cv,
            verbose=verbose,
            return_n_best_estimators=return_n_best_estimators,
            backend=backend,
            error_score=error_score,
            backend_params=backend_params,
            update_behaviour=update_behaviour,
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
        from skpro.survival.compose._reduce_cond_unc import ConditionUncensored

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

        params3 = {
            "estimator": ConditionUncensored(ResidualDouble(LinearRegression())),
            "cv": KFold(n_splits=4),
            "param_distributions": {"estimator__fit_intercept": [True, False]},
            "scoring": PinballLoss(),
            "error_score": "raise",
        }
        param_no_backend = {**param1, "backend": "None"}
        params = [param1, param2, params3, param_no_backend]

        from skpro.regression.online._refit import OnlineRefit

        params4 = {
            "estimator": OnlineRefit(ResidualDouble(LinearRegression())),
            "cv": KFold(n_splits=3),
            "param_distributions": {"estimator__fit_intercept": [True, False]},
            "scoring": CRPS(),
            "error_score": "raise",
            "update_behaviour": "inner_only",
        }

        params = [param1, param2, params3, params4]

        return params
