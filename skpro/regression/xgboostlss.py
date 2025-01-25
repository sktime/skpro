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
    }

    def __init__(
        self,
        dist="Normal",
        stabilization="None",
        response_fn="exp",
        loss_fn="nll",
        n_cpu="auto",
    ):
        self.dist = dist
        self.stabilization = stabilization
        self.response_fn = response_fn
        self.loss_fn = loss_fn
        self.n_cpu = n_cpu

        super().__init__()

        if n_cpu == "auto":
            import multiprocessing

            self._n_cpu = multiprocessing.cpu_count()
        else:
            self._n_cpu = n_cpu

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
        from xgboostlss.distributions import Gaussian
        from xgboostlss.model import XGBoostLSS

        self._y_cols = y.columns
        n_cpu = self._n_cpu

        dtrain = xgb.DMatrix(X, label=y, n_cpu=n_cpu, silent=True)

        xgblss = XGBoostLSS(
            Gaussian(
                stabilization="None",  
                response_fn="exp",      
                loss_fn="nll",
            )
        )

        param_dict = {
            "eta":              ["float", {"low": 1e-5,   "high": 1,     "log": True}],
            "max_depth":        ["int",   {"low": 1,      "high": 10,    "log": False}],
            "gamma":            ["float", {"low": 1e-8,   "high": 40,    "log": True}],
            "subsample":        ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
            "colsample_bytree": ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
            "min_child_weight": ["float", {"low": 1e-8,   "high": 500,   "log": True}],
            "booster":          ["categorical", ["gbtree"]],
        }

        opt_param = xgblss.hyper_opt(
            param_dict,
            dtrain,
            num_boost_round=100,
            # Number of boosting iterations.
            nfold=5,
            # Number of cv-folds.
            early_stopping_rounds=20,
            # Number of early-stopping rounds
            max_minutes=10,
            # Time budget in minutes,
            # i.e., stop study after the given number of minutes.
            n_trials=30,
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
        n_rounds = opt_params["opt_rounds"]
        del opt_params["opt_rounds"]

        # Train Model with optimized hyperparameters
        xgblss.train(
            opt_params,
            dtrain,
            num_boost_round=n_rounds
        )

        self.xgblss_ = xgblss
        return self

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

        dtest = xgb.DMatrix(X, n_cpu=n_cpu, silent=True)

        y_pred_xgblss = self.xgblss_.predict(dtest, pred_type="parameters")

        from skpro.distributions.normal import Normal

        y_pred = Normal(
            mu=y_pred_xgblss.iloc[:, [0]].values,  # mean is first column
            sigma=y_pred_xgblss.iloc[:, [1]].values,  # scale is second column
            index=index,
            columns=columns,
        )

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
        params0 = {}
        params1 = {}
