# skpro/regression/adapters/_statmixedml.py

from skpro.regression.base import BaseProbaRegressor


class _StatMixedMLMixin(BaseProbaRegressor):
    """Common mixin for StatMixedML distributional regressors (xgboostlss, lightgbmlss).

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

    n_cpu: int or str, optional, default="auto"
        Number of CPUs to use for parallel processing of data in ``lightgbmlss``.
        Default is "auto" which uses all available CPUs.

    num_boost_round: int, optional, default=100
        Number of boosting iterations.

    nfold: int, optional, default=5
        Number of folds in CV used for tuning.

    early_stopping_rounds: int, optional, default=20
        Number of early stopping round interval.
        Cross-Validation metric (average of validation metric computed over CV folds)
        needs to improve at least once every **early_stopping_rounds**
        round(s) to continue training.
        The last entry in the evaluation history will represent the best iteration.

    max_minutes: int, optional, default=10
        Time budget in minutes, i.e., stop study after the given number of minutes.

    n_trials: int, optional, default=30
        The number of trials in tuning.
        If this argument is set to None, there is no limitation on the number of trials.
        If set to 0, no tuning is done, and default parameters of LightGBMLSS are used.

    """

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
        **kwargs,
    ):
        self.dist = dist
        self.stabilization = stabilization
        self.response_fn = response_fn
        self.loss_fn = loss_fn
        self.n_cpu = n_cpu
        self.num_boost_round = num_boost_round
        self.nfold = nfold
        self.early_stopping_rounds = early_stopping_rounds
        self.max_minutes = max_minutes
        self.n_trials = n_trials

        # let concrete class tags be applied
        super().__init__(**kwargs)

        if n_cpu == "auto":
            import multiprocessing

            self._n_cpu = multiprocessing.cpu_count()
        else:
            self._n_cpu = n_cpu

        # conditionally require optuna for tuning
        if self.n_trials is not None and self.n_trials > 0:
            deps = self.get_tag("python_dependencies", tag_value_default=[])
            if "optuna" not in deps:
                deps = [*deps, "optuna"]
            self.set_tags(python_dependencies=deps)

    # shared helper for skpro distributions
    def _get_skpro_distr(self, distr):
        import importlib

        module_str = "skpro.distributions"
        object_str = distr

        module = importlib.import_module(module_str)
        return getattr(module, object_str)

    # shared parameter-name conversion
    def _get_skpro_val_dict(self, distr, df):
        name_map = {
            "Normal": {"mu": "loc", "sigma": "scale"},
            "Gamma": {"alpha": "concentration", "beta": "rate"},
            "Laplace": {"mu": "loc", "scale": "scale"},
            "LogNormal": {"mu": "loc", "sigma": "scale"},
            "TDistribution": {"mu": "loc", "sigma": "scale", "df": "df"},
            "Weibull": {"scale": "scale", "k": "concentration"},
            "Beta": {"alpha": "concentration1", "beta": "concentration0"},
        }
        map = name_map.get(distr, {})
        vals = {k: df.loc[:, [v]].values for k, v in map.items()}
        return vals

    def _hyper_opt(self, model, dtrain):
        """Run internal hyperparameter optimization.

        Uses the StatMixedML backend's own ``hyper_opt`` method.
        Dispatches parameter grids depending on backend (LightGBMLSS vs XGBoostLSS).
        """
        backend_module = model.__class__.__module__.split(".")[0].lower()

        if backend_module == "xgboostlss":
            # matches upstream XGBoostLSS examples
            param_dict = {
                "eta": ["float", {"low": 1e-5, "high": 1, "log": True}],
                "max_depth": ["int", {"low": 1, "high": 10, "log": False}],
                "gamma": ["float", {"low": 1e-8, "high": 40, "log": True}],
                "subsample": ["float", {"low": 0.2, "high": 1.0, "log": False}],
                "colsample_bytree": ["float", {"low": 0.2, "high": 1.0, "log": False}],
                "min_child_weight": ["float", {"low": 1e-8, "high": 500, "log": True}],
                "booster": ["categorical", ["gbtree"]],
            }
        elif backend_module == "lightgbmlss":
            # compact, generic LightGBM-style param grid
            param_dict = {
                "learning_rate": ["float", {"low": 1e-3, "high": 0.3, "log": True}],
                "num_leaves": ["int", {"low": 16, "high": 256, "log": False}],
                "max_depth": ["int", {"low": 1, "high": 16, "log": False}],
                "min_child_samples": ["int", {"low": 10, "high": 200, "log": False}],
                "subsample": ["float", {"low": 0.2, "high": 1.0, "log": False}],
                "colsample_bytree": ["float", {"low": 0.2, "high": 1.0, "log": False}],
            }
        else:
            raise RuntimeError(
                f"Unknown StatMixedML backend '{backend_module}' in _hyper_opt"
            )

        opt_param = model.hyper_opt(
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
        params1 = {"stabilization": "L2", "loss_fn": "crps", "max_minutes": 1}
        params2 = {"dist": "Gamma", "max_minutes": 1}
        params3 = {"dist": "Weibull", "max_minutes": 1}
        params4 = {"dist": "TDistribution", "max_minutes": 1}
        params5 = {"dist": "Laplace", "max_minutes": 1}
        params6 = {"n_trials": 0, "max_minutes": 1}
        params7 = {"dist": "Beta", "max_minutes": 1}
        return [params0, params1, params2, params3, params4, params5, params6, params7]
