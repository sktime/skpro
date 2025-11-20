"""Interface for lightgbmlss probabilistic regressor."""

from skpro.regression.base import BaseProbaRegressor


class LightGBMLSS(BaseProbaRegressor):
    """Interface to lightgbmlss regressor from the lightgbmlss package.

    Direct interface to ``LightGBMLSS`` from ``lightgbmlss`` package by ``StatMixedML``.

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

    _tags = {
        # packaging info
        # --------------
        "authors": ["StatMixedML", "EchoDel", "fkiraly"],
        "python_dependencies": ["lightgbmlss", "lightgbm"],
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

    def _get_lgblss_distr(self, distr):
        """Get lightgbmlss distribution object from string.

        Parameters
        ----------
        distr : str
            Distribution name, in skpro, as in self.dist
        """
        import importlib

        SKPRO_TO_LGBLSS = {
            "Normal": "Gaussian",
            "TDistribution": "StudentT",
        }
        distr = SKPRO_TO_LGBLSS.get(distr, distr)

        module_str = "lightgbmlss.distributions." + distr
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
        """Convert lightgbmlss parameters to skpro distribution.

        Parameters
        ----------
        distr : str
            Distribution name, in skpro, as in self.dist
        df : pd.DataFrame
            DataFrame of parameters as returned by predict, in lightgbmlss.
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
        import lightgbm as lgb
        from lightgbmlss.model import LightGBMLSS as _LightGBMLSS

        self._y_cols = y.columns

        # lightgbmlss examples use 1D numpy array for label
        y_label = y.iloc[:, 0].values

        dtrain = lgb.Dataset(X, label=y_label)

        lgblss_distr = self._get_lgblss_distr(self.dist)

        lgblss = _LightGBMLSS(
            lgblss_distr(
                stabilization=self.stabilization,
                response_fn=self.response_fn,
                loss_fn=self.loss_fn,
            )
        )

        if self.n_trials == 0:
            opt_params = {}  # empty dict, use default parameters
            n_rounds = self.num_boost_round
        else:
            opt_params = self._hyper_opt(lgblss, dtrain)
            # LightGBMLSS hyper_opt returns "opt_rounds" key for best iteration
            n_rounds = opt_params.pop("opt_rounds", self.num_boost_round)

        # Train Model with optimized hyperparameters
        lgblss.train(opt_params, dtrain, num_boost_round=n_rounds)

        self.lgblss_ = lgblss
        return self

    def _hyper_opt(self, lgblss, dtrain):
        """Run internal hyperparameter optimization.

        Uses ``lightgbmlss.hyper_opt`` method to run hyperparameter optimization.

        Parameters
        ----------
        lgblss : lightgbmlss.model.LightGBMLSS instance
            lightgbmlss model instance, as created in ``_fit`` method
        dtrain : lightgbm.Dataset
            training data, as created in ``_fit`` method

        Returns
        -------
        opt_params : dict
            dictionary of hyperparameters to be passed to lightgbmlss
        """
        param_dict = {
            "eta": ["float", {"low": 1e-5, "high": 1, "log": True}],
            "max_depth": ["int", {"low": 1, "high": 10, "log": False}],
            "num_leaves": ["int", {"low": 31, "high": 255, "log": False}],
            "min_data_in_leaf": ["int", {"low": 10, "high": 200, "log": False}],
            "min_gain_to_split": ["float", {"low": 1e-8, "high": 40, "log": False}],
            "min_sum_hessian_in_leaf": [
                "float",
                {"low": 1e-8, "high": 500, "log": True},
            ],
            "subsample": ["float", {"low": 0.2, "high": 1.0, "log": False}],
            "feature_fraction": ["float", {"low": 0.2, "high": 1.0, "log": False}],
            "boosting": ["categorical", ["gbdt"]],
        }

        opt_param = lgblss.hyper_opt(
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
            # Controls the verbosity of the trial,
            # i.e., user can silence the outputs of the trial.
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
        index = X.index
        y_cols = self._y_cols
        columns = y_cols

        # lightgbmlss.predict takes raw feature matrix / DataFrame
        y_pred_lgblss = self.lgblss_.predict(X, pred_type="parameters")

        skpro_distr = self._get_skpro_distr(self.dist)
        skpro_distr_vals = self._get_skpro_val_dict(self.dist, y_pred_lgblss)

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
        return [params0, params1, params2, params3, params4, params5, params6, params7]

