"""Interface adapter for MAPIE regressor."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]  # interface only. MAPIE authors in mapie package

import numpy as np
import pandas as pd

from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class MapieRegressor(BaseProbaRegressor):
    """MAPIE probabilistic regressor, conformity score based prediction intervals.

    Direct interface to ``mapie.regression.regression.MapieRegressor`` from the
    ``mapie`` package.

    Uses jackknife+ to estimate prediction intervals on a per-sample basis.

    Any (non-probabilistic) sklearn regressor can be used as the base regressor,
    ``MapieRegressor`` will add prediction intervals.

    Parameters
    ----------
    estimator: sklearn regressor, default = ``sklearn.linear_model.LinearRegression``
        regressor with scikit-learn compatible API
        (requires ``fit``, ``predict``, ``get_params``, ``set_params``).
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

    method: str, default = "plus"
        Method to choose for prediction interval estimates.

        - ``"naive"``, based on training set conformity scores,
        - ``"base"``, based on validation sets conformity scores,
        - ``"plus"``, based on validation conformity scores and
          testing predictions,
        - ``"minmax"``, based on validation conformity scores and
          testing predictions (min/max among cross-validation clones).

    cv: int, str, or sklearn splitter, default = 5-fold cross-validation
        Cross-validation strategy for computing conformity scores.

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to ``-1``, equivalent to
          ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
          Main variants are:
            - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
            - ``sklearn.model_selection.KFold`` (cross-validation),
            - ``subsample.Subsample`` object (bootstrap).
        - ``"split"``, does not involve cross-validation but a division
          of the data into training and calibration subsets. The splitter
          used is the following: ``sklearn.model_selection.ShuffleSplit``.
          ``method`` parameter is set to ``"base"``.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already,
          and the ``method`` parameter is set to ``"base"``.
          All data provided in the ``fit`` method is then used
          for computing conformity scores only.
          At prediction time, quantiles of these conformity scores are used
          to provide a prediction interval with fixed width.
          The user has to take care manually that data for model fitting and
          conformity scores estimate are disjoint.

        By default ``None``.

    test_size: int, float, or None (default).
        Test size for the ``"split"`` cross-validation strategy.
        Ignored unless ``cv`` is ``"split"``.

        If ``float``, must be between ``0.0`` and ``1.0`` and represents the
        proportion of the dataset to include in the test split.
        If ``int``, represents the absolute number of test samples.
        If ``None``, set to ``0.1``.

    n_jobs: int, optional, default=None
        Number of jobs for parallel processing using ``joblib``
        via the "locky" backend.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For ``n_jobs`` below ``-1``, ``(n_cpus + 1 - n_jobs)`` are used.
        ``None`` is a marker for `unset` that will be interpreted as
        ``n_jobs=1`` (sequential execution).

    agg_function: str, one of "mean" (default), "median", or None.
        Determines how to aggregate predictions from perturbed models, both at
        training and prediction time.

        If ``None``, it is ignored except if ``cv`` class is ``Subsample``,
        in which case an error is raised.
        If ``"mean"`` or ``"median"``, returns the mean or median of the
        predictions computed from the out-of-folds models.
        Note: if you plan to set the ``ensemble`` argument to ``True`` in the
        ``predict`` method, you have to specify an aggregation function.
        Otherwise an error would be raised.

        The Jackknife+ interval can be interpreted as an interval around the
        median prediction, and is guaranteed to lie inside the interval,
        unlike the single estimator predictions.

        When the cross-validation strategy is ``Subsample`` (i.e. for the
        Jackknife+-after-Bootstrap method), this function is also used to
        aggregate the training set in-sample predictions.

        If ``cv`` is ``"prefit"`` or ``"split"``, ``agg_function`` is ignored.

    verbose: int, optional, default=0
        The verbosity level, used with joblib for multiprocessing.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

    conformity_score: ``ConformityScore``, optional, default = absolute conformity score
        mapie ConformityScore descendant instance, from ``mapie.conformity_scores``.
        It defines the link between the observed values, the predicted ones
        and the conformity scores. For instance, the default ``None`` value
        correspondonds to a conformity score which assumes
        y_obs = y_pred + conformity_score.

        - ``None``, to use the default ``AbsoluteConformityScore`` conformity
          score
        - ConformityScore: any ``ConformityScore`` class

    ensemble: bool, optional, default=False
        Boolean determining whether predictions are ensembled or not.
        Ignored if ``cv`` is ``"prefit"`` or ``"split"``.

        If ``False``, predictions areof the model trained on the full training set.
        If ``True``, predictions from perturbed models are aggregated by
        the aggregation function specified in the ``agg_function``attribute.

    random_state: None (default), ``int``, or ``RandomState`` instance
        Pseudo random number generator state used for random sampling.
        Pass an int for reproducible output across multiple function calls.
        If None, no random state is used.

    Attributes
    ----------
    estimator_: EnsembleRegressor
        fitted mapie ensemble regressor

    conformity_scores_: ArrayLike of shape (n_samples_train,)
        Conformity scores between ``y_train`` and ``y_pred``.

    Example
    -------
    >>> from skpro.regression.mapie import MapieRegressor  # doctest: +SKIP
    >>> from sklearn.ensemble import RandomForestRegressor  # doctest: +SKIP
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>> from sklearn.model_selection import train_test_split  # doctest: +SKIP
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)  # doctest: +SKIP
    >>>
    >>> reg_tabular = RandomForestRegressor()  # doctest: +SKIP
    >>>
    >>> reg_proba = MapieRegressor(reg_tabular)  # doctest: +SKIP
    >>> reg_proba.fit(X_train, y_train)  # doctest: +SKIP
    MapieRegressor(...)
    >>> y_pred_int = reg_proba.predict_interval(X_test)  # doctest: +SKIP
    >>> y_pred_dist = reg_proba.predict_proba(X_test)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly"],
        "python_dependencies": ["mapie"],
        # estimator tags
        # --------------
        "capability:missing": True,
    }

    def __init__(
        self,
        estimator=None,
        method="plus",
        cv=None,
        test_size=None,
        n_jobs=None,
        agg_function="mean",
        verbose=0,
        conformity_score=None,
        random_state=None,
    ):
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.agg_function = agg_function
        self.verbose = verbose
        self.conformity_score = conformity_score
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        # construct mapie regressor
        from mapie.regression.regression import MapieRegressor

        PARAMS_TO_FORWARD = [
            "estimator",
            "method",
            "cv",
            "test_size",
            "n_jobs",
            "agg_function",
            "verbose",
            "conformity_score",
            "random_state",
        ]

        params = self.get_params(deep=False)
        params = {k: v for k, v in params.items() if k in PARAMS_TO_FORWARD}
        mapie_est_ = MapieRegressor(**params)

        # remember y columns for predict
        self._y_cols = y.columns

        # coerce y to numpy array
        if len(y.columns) == 1:
            y = y.to_numpy().flatten()

        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        # fit mapie regressor and save to self
        mapie_est_.fit(X, y)

        self.estimator_mapie_ = mapie_est_

        # forward fitted attributes to self
        FITTED_PARAMS_TO_FORWARD = ["estimator_", "conformity_scores_"]

        for param in FITTED_PARAMS_TO_FORWARD:
            setattr(self, param, getattr(self.estimator_mapie_, param))

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted" = self.is_fitted=True

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            labels predicted for `X`
        """
        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        # predict with mapie regressor
        y_pred_mapie = self.estimator_mapie_.predict(X)

        # format output as pandas DataFrame with correct indices
        index = X.index
        columns = self._y_cols
        y_pred = pd.DataFrame(y_pred_mapie, index=index, columns=columns)

        return y_pred

    def _predict_interval(self, X, coverage):
        """Compute/return interval predictions.

        private _predict_interval containing the core logic,
            called from predict_interval and default _predict_quantiles

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for
        coverage : guaranteed list of float of unique values
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from ``y`` in fit,
            second level coverage fractions for which intervals were computed,
            in the same order as in input `coverage`.
            Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is equal to row index of ``X``.
            Entries are lower/upper bounds of interval predictions,
            for var in col index, at nominal coverage in second col index,
            lower/upper depending on third col index, for the row index.
            Upper/lower interval end are equivalent to
            quantile predictions at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        cov_arr = np.array(coverage)
        mapie_alpha = 1 - cov_arr

        mapie_pred_int = self.estimator_mapie_.predict(X, alpha=mapie_alpha)[1]

        index = X.index
        columns = pd.MultiIndex.from_product(
            [self._y_cols, coverage, ["lower", "upper"]],
        )

        values = np.reshape(mapie_pred_int, (len(index), -1), order="F")
        pred_int = pd.DataFrame(values, index=index, columns=columns)

        return pred_int

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.ensemble import RandomForestRegressor

        params1 = {}

        params2 = {
            "estimator": RandomForestRegressor(),
            "method": "base",
            "cv": 2,
            "test_size": 0.2,
            "agg_function": "median",
            "conformity_score": None,
            "random_state": 42,
        }

        return [params1, params2]
