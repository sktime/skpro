"""Interface for xgboostlss probabilistic regressor."""

import warnings

from skpro.regression.base import BaseProbaRegressor


class XGBoostLSS(BaseProbaRegressor):
    """Interface to xgboostlss regerssor from the xgboostlss package.

    Direct interface to ``XGBoostLSS`` from ``xgboostlss`` package by ``StatMixedML``.

    Parameters
    ----------
    dist: str, optional, default="Normal"
        Form of predictive distribution, strings are same as in skpro.

        Valid options are:

        * "Normal": Normal distribution.
        * "Gamma": Gamma distribution.
        * "Laplace": Laplace distribution.
        * "LogNormal": LogNormal distribution.
        * "TDistribution": Student's T distribution.
        * "Weibull": Weibull distribution.
        * "Beta": Beta distribution.

    stabilization: str, optional, default="None"
        Stabilization method for the Gradient and Hessian.
        Options are "None", "MAD", "L2".

    response_fn: str, optional, default="exp"
        Response function for transforming the distributional parameters to the
        support of the distribution. Options are "exp" (exponential) or
        "softplus" (softplus).

    loss_fn: str, optional, default="nll"
        Loss function used in tuning and fitting.
        Options are "nll" (negative log-likelihood)
        or "crps" (continuous ranked probability score).

    n_jobs: int, optional, default=None
        Number of CPUs to use for parallel processing of data in ``xgboostlss``.
        If None, then the value of n_cpu is used.

    n_cpu: int or str, optional, default="auto"
        Number of CPUs to use for parallel processing of data in ``xgboostlss``.
        Default is "auto" which uses all available CPUs. If an integer is given,
        that number of CPUs is used. If both n_cpu and n_jobs are set, an error
        is raised. `n_jobs` should be preferred to `n_cpu` for sklearn compatibility.

    n_estimators: int, optional, default=None
        Number of boosting iterations. Alternative to num_boost_round. This is
        the sklearn standard library parameter name.

    num_boost_round: int, optional, default=100
        Number of boosting iterations. Ignored if n_estimators is set. This is
        the XGBoost standard library parameter name.

    nfold: int, optional, default=5
        Number of folds in CV used for tuning. Ignored if n_trials=0.

    early_stopping_rounds: int, optional, default=20
        Number of early stopping round interval.
        Cross-Validation metric (average of validation metric computed over CV folds)
        needs to improve at least once every **early_stopping_rounds**
        round(s) to continue training.
        The last entry in the evaluation history will represent the best iteration.
        Ignored if n_trials=0.

    max_minutes: int, optional, default=10
        Time budget in minutes, i.e., stop study after the given number of minutes.

    n_trials: int, optional, default=30
        The number of trials in tuning.
        If this argument is set to None, there is no limitation on the number of trials.
        If set to 0, no tuning is done, and default parameters of XGBoostLSS are used.

    **kwargs: optional
        Keyword arguments of xgboost parameters to pass to the model.
        Used only if n_trials=0 (i.e., no hyperparameter optimization).
        See https://xgboost.readthedocs.io/en/stable/parameter.html for valid params.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["StatMixedML", "EchoDel", "fkiraly"],
        # StatMixedML, EchoDel for the original xgboostlss package
        "maintainers": ["fkiraly"],
        "python_dependencies": ["xgboostlss"],  # PEP 440 python dependencies specifier,
        #
        # estimator tags
        # --------------
        "capability:multioutput": False,  # can the estimator handle multi-output data?
        "capability:missing": True,  # can the estimator handle missing data?
        "X_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit, _predict
        "y_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit
        # CI and test flags
        # -----------------
        "tests:vm": True,  # requires its own test VM to run
    }

    _xgb_params = [
        "max_depth",
        "max_leaves",
        "max_bin",
        "grow_policy",
        "eta",
        "verbosity",
        "booster",
        "tree_method",
        "gamma",
        "min_child_weight",
        "max_delta_step",
        "subsample",
        "sampling_method",
        "colsample_bytree",
        "colsample_bylevel",
        "colsample_bynode",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
        "random_state",
        "num_parallel_tree",
        "monotone_constraints",
        "interaction_constraints",
        "importance_type",
        "device",
        "validate_parameters",
        "feature_types",
        "feature_weights",
        "max_cat_to_onehot",
        "max_cat_threshold",
        "multi_strategy",
        "eval_metric",
    ]

    def __init__(
        self,
        dist="Normal",
        stabilization="None",
        response_fn="exp",
        loss_fn="nll",
        n_cpu="auto",
        num_boost_round=100,
        nfold=5,
        early_stopping_rounds=20,
        max_minutes=10,
        n_trials=30,
        max_depth=None,
        max_leaves=None,
        max_bin=None,
        grow_policy=None,
        eta=None,
        learning_rate=None,
        n_estimators=None,
        verbosity=None,
        booster=None,
        tree_method=None,
        n_jobs=None,
        gamma=None,
        min_child_weight=None,
        max_delta_step=None,
        subsample=None,
        sampling_method=None,
        colsample_bytree=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        reg_alpha=None,
        reg_lambda=None,
        scale_pos_weight=None,
        random_state=None,
        num_parallel_tree=None,
        monotone_constraints=None,
        interaction_constraints=None,
        importance_type=None,
        device=None,
        validate_parameters=None,
        feature_types=None,
        feature_weights=None,
        max_cat_to_onehot=None,
        max_cat_threshold=None,
        multi_strategy=None,
        # set by xgboostlss internally
        eval_metric=None,
        callbacks=None,
    ):
        # distributional learning parameters:
        self.dist = dist
        self.stabilization = stabilization
        self.response_fn = response_fn
        # paired parameters:
        # eta in XGBoost, learning_rate in sklearn
        self.eta_ = eta
        self.learning_rate = learning_rate
        # nthreads in XGBoost, n_jobs in sklearn, n_cpu?
        self.n_jobs_ = n_jobs
        self.n_cpu = n_cpu
        # num_boost_round and n_estimators handling
        self.num_boost_round_ = num_boost_round
        self.n_estimators = n_estimators
        # xgboost training parameters:
        # loss_fn is set internally in XGBoostLSS
        self.loss_fn = loss_fn
        self.nfold = nfold
        self.early_stopping_rounds = early_stopping_rounds
        # hyperopt only parameters:
        self.max_minutes = max_minutes
        self.n_trials = n_trials
        # xgboost parameters:
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.grow_policy = grow_policy
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.sampling_method = sampling_method
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.device = device
        self.validate_parameters = validate_parameters
        self.feature_types = feature_types
        self.feature_weights = feature_weights
        self.max_cat_to_onehot = max_cat_to_onehot
        self.max_cat_threshold = max_cat_threshold
        self.multi_strategy = multi_strategy
        self.eval_metric = eval_metric
        # xgboostlss overrides base_score internally when using hyperopt
        self.callbacks = callbacks

        super().__init__()

    def _get_xgblss_distr(self, distr):
        """Get xgboostlss distribution object from string.

        Parameters
        ----------
        distr : str
            Distribution name, in skpro, as in self.dist
        """
        import importlib

        SKPRO_TO_XGBLSS = {
            "Normal": "Gaussian",
            "TDistribution": "StudentT",
        }

        distr = SKPRO_TO_XGBLSS.get(distr, distr)

        module_str = "xgboostlss.distributions." + distr
        object_str = distr

        module = importlib.import_module(module_str)
        return getattr(module, object_str)

    def _get_skpro_distr(self, distr):
        """Get skpro distribution object from string.

        Parameters
        ----------
        distr : str
            Distribution name, in skpro, as in self.dist
        """
        import importlib

        module_str = "skpro.distributions"
        object_str = distr

        module = importlib.import_module(module_str)
        return getattr(module, object_str)

    def _get_skpro_val_dict(self, distr, df):
        """Convert xgboostlss parameters to skpro distribution.

        `{"skpro_param": "xgboostlss_param"}`

        Parameters
        ----------
        distr : str
            Distribution name, in skpro, as in self.dist
        df : pd.DataFrame
            DataFrame of parameters as returned by predict, in xgboostlss.
        """
        name_map = {
            "Normal": {"mu": "loc", "sigma": "scale"},
            "Gamma": {"alpha": "concentration", "beta": "rate"},
            "Laplace": {"mu": "loc", "scale": "scale"},
            "LogNormal": {"mu": "loc", "sigma": "scale"},
            "TDistribution": {"mu": "loc", "sigma": "scale", "df": "df"},
            "Weibull": {"scale": "scale", "k": "concentration"},
            "Beta": {"alpha": "concentration1", "beta": "concentration0"},
            "Logisitic": {"mu": "loc", "scale": "scale"},
        }

        map = name_map.get(distr, {})
        vals = {k: df.loc[:, [v]].values for k, v in map.items()}
        return vals

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
        import xgboost as xgb
        from xgboostlss.model import XGBoostLSS

        # Set _n_jobs
        if self.n_jobs_ is not None:
            if self.n_cpu != "auto":
                raise ValueError("Cannot set both n_cpu and n_jobs")
            self._n_jobs = self.n_jobs_
        else:
            if self.n_cpu == "auto":
                import multiprocessing

                self._n_jobs = multiprocessing.cpu_count()
            else:
                self._n_jobs = self.n_cpu

        # Handle n_estimators and num_boost_round
        if self.n_estimators is not None and self.num_boost_round_ != 100:
            raise ValueError("Cannot set both n_estimators and num_boost_round")
        else:
            self.num_boost_round = self.n_estimators or self.num_boost_round_

        if self.learning_rate is not None and self.eta_ is not None:
            raise ValueError("Cannot set both learning_rate and eta")
        else:
            self.eta = self.learning_rate or self.eta_

        self._y_cols = y.columns
        dtrain = xgb.DMatrix(X, label=y, nthread=self._n_jobs, silent=True)

        xgblss_distr = self._get_xgblss_distr(self.dist)

        xgblss = XGBoostLSS(
            xgblss_distr(
                stabilization=self.stabilization,
                response_fn=self.response_fn,
                loss_fn=self.loss_fn,
            )
        )

        # Collect XGBoost params
        self.xgb_params = dict()
        for k in self._xgb_params:
            attr = getattr(self, k)
            if attr is not None:
                self.xgb_params[k] = getattr(self, k)

        if self.n_trials == 0:
            # use user-defined hyperparameters
            xgb_params = self.xgb_params
            n_rounds = self.num_boost_round
            train_kwargs = {}

            if self.callbacks is not None:
                train_kwargs["callbacks"] = self.callbacks
        else:
            # get optimized hyperparameters
            xgb_params = self._hyper_opt(xgblss, dtrain)
            n_rounds = xgb_params.pop("num_boost_round", self.num_boost_round)
            train_kwargs = {}

            if self.callbacks is not None:
                warnings.warn(
                    "Callbacks are not supported in hyperparameter optimization. "
                    "Ignoring callbacks.",
                    UserWarning,
                    stacklevel=2,
                )

        xgblss.train(xgb_params, dtrain, num_boost_round=n_rounds, **train_kwargs)

        self.xgblss_ = xgblss
        return self

    def _hyper_opt(self, xgblss, dtrain):
        """Run internal hyperparameter optimization.

        Uses ``xgboostlss.hyper_opt`` function to run hyperparameter optimization.

        Parameters
        ----------
        xgblss : xgboostlss.model.XGBoostLSS instance
            xgboostlss model instance, as created in ``_fit`` method
        dtrain : xgboost.DMatrix
            training data, as created in ``_fit`` method

        Returns
        -------
        opt_params : dict
            dictionary of hyperparameters to be passed to xgboostlss
        """
        param_dict = {
            "eta": ["float", {"low": 1e-5, "high": 1, "log": True}],
            "max_depth": ["int", {"low": 1, "high": 10, "log": False}],
            "gamma": ["float", {"low": 1e-8, "high": 40, "log": True}],
            "subsample": ["float", {"low": 0.2, "high": 1.0, "log": False}],
            "colsample_bytree": ["float", {"low": 0.2, "high": 1.0, "log": False}],
            "min_child_weight": ["float", {"low": 1e-8, "high": 500, "log": True}],
            "booster": ["categorical", ["gbtree"]],
        }

        opt_param = xgblss.hyper_opt(
            param_dict,
            dtrain,
            num_boost_round=self.num_boost_round,
            # Number of boosting iterations.
            nfold=self.nfold,
            # Number of cv-folds.
            early_stopping_rounds=self.early_stopping_rounds,
            # Number of early-stopping rounds
            max_minutes=self.max_minutes,
            # Time budget in minutes,
            # i.e., stop study after the given number of minutes.
            n_trials=self.n_trials,
            # The number of trials. If this argument is set to None,
            # there is no limitation on the number of trials.
            silence=True,
            # Controls the verbosity of the trail,
            # i.e., user can silence the outputs of the trail.
            seed=123,
            # Seed used to generate cv-folds.
            hp_seed=123,
            # Seed for random number generator used
            # in the Bayesian hyperparameter search.
        )

        opt_params = opt_param.copy()
        return opt_params

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y_pred : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        import xgboost as xgb

        index = X.index
        y_cols = self._y_cols
        columns = y_cols

        dtest = xgb.DMatrix(X, nthread=self._n_jobs, silent=True)

        y_pred_xgblss = self.xgblss_.predict(dtest, pred_type="parameters")

        skpro_distr = self._get_skpro_distr(self.dist)
        skpro_distr_vals = self._get_skpro_val_dict(self.dist, y_pred_xgblss)

        y_pred = skpro_distr(**skpro_distr_vals, index=index, columns=columns)

        return y_pred

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
        params0 = {"max_minutes": 1}
        params1 = {
            "stabilization": "L2",
            "loss_fn": "crps",
            "max_minutes": 1,
        }
        params2 = {"dist": "Gamma", "max_minutes": 1}
        params3 = {"dist": "Weibull", "max_minutes": 1}
        params4 = {"dist": "TDistribution", "max_minutes": 1}
        params5 = {"dist": "Laplace", "max_minutes": 1}
        params6 = {"n_trials": 0, "max_minutes": 1}
        params7 = {"dist": "Beta", "max_minutes": 1}
        params8 = {
            "n_trials": 0,
            "max_minutes": 1,
            "xgb_params": {"eta": 0.1, "max_depth": 3},
        }
        return [
            params0,
            params1,
            params2,
            params3,
            params4,
            params5,
            params6,
            params7,
            params8,
        ]
