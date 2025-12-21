"""Interface for xgboostlss probabilistic regressor."""

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

    n_cpu: int or str, optional, default="auto"
        Number of CPUs to use for parallel processing of data in ``xgboostlss``.
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
        If set to 0, no tuning is done, and default parameters of XGBoostLSS are used.
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
        "tests:python_dependencies": ["optuna", "optuna-integration"],
    }

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

        super().__init__()

        if n_cpu == "auto":
            import multiprocessing

            self._n_cpu = multiprocessing.cpu_count()
        else:
            self._n_cpu = n_cpu

        # If n_trials is not zero, optuna is required for hyperparameter optimization
        if n_trials != 0:
            self.set_tags(**{"python_dependencies": ["xgboostlss", "optuna"]})

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

        self._y_cols = y.columns
        n_cpu = self._n_cpu

        dtrain = xgb.DMatrix(X, label=y, nthread=n_cpu, silent=True)

        xgblss_distr = self._get_xgblss_distr(self.dist)

        xgblss = XGBoostLSS(
            xgblss_distr(
                stabilization="None",
                response_fn="exp",
                loss_fn="nll",
            )
        )

        if self.n_trials == 0:
            opt_params = {}  # empty dict, use default parameters
            n_rounds = self.num_boost_round
        else:
            opt_params = self._hyper_opt(xgblss, dtrain)
            n_rounds = opt_params.pop("n_estimators", self.num_boost_round)

        # Train Model with optimized hyperparameters
        xgblss.train(opt_params, dtrain, num_boost_round=n_rounds)

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

        n_cpu = self._n_cpu

        index = X.index
        y_cols = self._y_cols
        columns = y_cols

        dtest = xgb.DMatrix(X, nthread=n_cpu, silent=True)

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
        params0 = {"max_minutes": 1, "n_trials": 2}
        params1 = {
            "stabilization": "L2",
            "loss_fn": "crps",
            "max_minutes": 1,
            "n_trials": 2,
        }
        params2 = {"dist": "Gamma", "max_minutes": 1, "n_trials": 2}
        params3 = {"dist": "Weibull", "max_minutes": 1, "n_trials": 2}
        params4 = {"dist": "TDistribution", "max_minutes": 1, "n_trials": 2}
        params5 = {"dist": "Laplace", "max_minutes": 1, "n_trials": 2}
        params6 = {"n_trials": 0, "max_minutes": 1}
        params7 = {"dist": "Beta", "max_minutes": 1, "n_trials": 2}
        return [params0, params1, params2, params3, params4, params5, params6, params7]
