"""Interface for lightgbmlss probabilistic regressor."""

import warnings

from skpro.regression.base import BaseProbaRegressor

__all__ = ["LightGBMLSS"]


class LightGBMLSS(BaseProbaRegressor):
    """Interface to lightgbmlss regressor from the lightgbmlss package.

    Direct interface to ``LightGBMLSS`` from the ``lightgbmlss`` package by
    ``StatMixedML``.

    Parameters
    ----------
    dist : str, optional, default="Normal"
        Form of predictive distribution, strings are same as in skpro.

        Valid options are:

        * "Normal": Normal distribution.
        * "Gamma": Gamma distribution.
        * "Laplace": Laplace distribution.
        * "LogNormal": LogNormal distribution.
        * "TDistribution": Student's T distribution.
        * "Weibull": Weibull distribution.
        * "Beta": Beta distribution.
        * "Logistic": Logistic distribution.
    stabilization : str, optional, default="None"
        Stabilization method for the Gradient and Hessian.
        Options are "None", "MAD", "L2".
    response_fn : str, optional, default="exp"
        Response function for transforming the distributional parameters to the
        support of the distribution. Options are "exp" or "softplus".
    loss_fn : str, optional, default="nll"
        Loss function used in tuning and fitting.
        Options are "nll" (negative log-likelihood)
        or "crps" (continuous ranked probability score).
    initialize : bool, optional, default=True
        Whether to initialize the distribution parameters.
    n_jobs : int, optional, default=None
        Number of CPUs to use for parallel processing.
        If None, then the value of ``num_threads`` is used.
    num_threads : int or str, optional, default="auto"
        Number of threads to use for LightGBM training. If "auto", all available
        CPUs are used. If both ``num_threads`` and ``n_jobs`` are set, an error
        is raised. ``n_jobs`` should be preferred for sklearn compatibility.
    n_estimators : int, optional, default=None
        Number of boosting iterations. Alternative to ``num_boost_round``.
    num_boost_round : int, optional, default=100
        Number of boosting iterations. Ignored if ``n_estimators`` is set.
    nfold : int, optional, default=5
        Number of folds in CV used for tuning. Ignored if ``n_trials=0``.
    early_stopping_rounds : int, optional, default=20
        Number of rounds without improvement before stopping hyperparameter search.
        Ignored if ``n_trials=0``.
    max_minutes : int, optional, default=10
        Time budget in minutes for hyperparameter optimization.
    n_trials : int, optional, default=30
        Number of trials in tuning. If set to 0, no tuning is done and the
        provided LightGBM parameters are used directly.
    explicitly_named_kwargs
        Keyword arguments of LightGBM parameters to pass to the model when
        ``n_trials=0``.
    """

    _tags = {
        "authors": ["StatMixedML", "fkiraly"],
        "maintainers": ["fkiraly"],
        "python_dependencies": ["lightgbmlss"],
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        "tests:vm": True,
        "tests:python_dependencies": ["optuna", "optuna-integration"],
    }

    def __init__(
        self,
        dist="Normal",
        stabilization="None",
        response_fn="exp",
        loss_fn="nll",
        initialize=True,
        num_threads="auto",
        num_boost_round=100,
        nfold=5,
        early_stopping_rounds=20,
        max_minutes=10,
        n_trials=30,
        boosting=None,
        learning_rate=None,
        num_leaves=None,
        max_depth=None,
        min_data_in_leaf=None,
        min_sum_hessian_in_leaf=None,
        feature_fraction=None,
        feature_fraction_bynode=None,
        bagging_fraction=None,
        bagging_freq=None,
        lambda_l1=None,
        lambda_l2=None,
        min_gain_to_split=None,
        max_bin=None,
        extra_trees=None,
        linear_tree=None,
        data_sample_strategy=None,
        n_estimators=None,
        n_jobs=None,
        callbacks=None,
    ):
        self.dist = dist
        self.stabilization = stabilization
        self.response_fn = response_fn
        self.loss_fn = loss_fn
        self.initialize = initialize

        self.n_jobs = n_jobs
        self.num_threads = num_threads
        self.num_boost_round = num_boost_round
        self.n_estimators = n_estimators
        self.nfold = nfold
        self.early_stopping_rounds = early_stopping_rounds
        self.max_minutes = max_minutes
        self.n_trials = n_trials

        self.boosting = boosting
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.feature_fraction = feature_fraction
        self.feature_fraction_bynode = feature_fraction_bynode
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.max_bin = max_bin
        self.extra_trees = extra_trees
        self.linear_tree = linear_tree
        self.data_sample_strategy = data_sample_strategy
        self.callbacks = callbacks

        super().__init__()

        if n_trials != 0:
            self.set_tags(
                **{
                    "python_dependencies": [
                        "lightgbmlss",
                        "optuna",
                        "optuna-integration",
                    ]
                }
            )

        self._lgb_params = [
            "boosting",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_data_in_leaf",
            "min_sum_hessian_in_leaf",
            "feature_fraction",
            "feature_fraction_bynode",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "min_gain_to_split",
            "max_bin",
            "extra_trees",
            "linear_tree",
            "data_sample_strategy",
            "num_threads",
        ]

    def _get_lgblss_distr(self, distr):
        """Get lightgbmlss distribution class from a skpro distribution name."""
        import importlib

        skpro_to_lgblss = {
            "Normal": "Gaussian",
            "TDistribution": "StudentT",
        }

        distr = skpro_to_lgblss.get(distr, distr)

        module = importlib.import_module(f"lightgbmlss.distributions.{distr}")
        return getattr(module, distr)

    def _get_skpro_distr(self, distr):
        """Get skpro distribution class from string."""
        import importlib

        module = importlib.import_module("skpro.distributions")
        return getattr(module, distr)

    def _get_skpro_val_dict(self, distr, df):
        """Convert lightgbmlss parameters to skpro distribution parameters."""
        name_map = {
            "Normal": {"mu": "loc", "sigma": "scale"},
            "Gamma": {"alpha": "concentration", "beta": "rate"},
            "Laplace": {"mu": "loc", "scale": "scale"},
            "LogNormal": {"mu": "loc", "sigma": "scale"},
            "TDistribution": {"mu": "loc", "sigma": "scale", "df": "df"},
            "Weibull": {"scale": "scale", "k": "concentration"},
            "Beta": {"alpha": "concentration1", "beta": "concentration0"},
            "Logistic": {"mu": "loc", "scale": "scale"},
        }

        param_map = name_map.get(distr, {})
        return {k: df.loc[:, [v]].values for k, v in param_map.items()}

    def _fit(self, X, y):
        """Fit regressor to training data."""
        import lightgbm as lgb
        from lightgbmlss.model import LightGBMLSS as _LightGBMLSS

        if self.n_jobs is not None:
            if self.num_threads != "auto":
                raise ValueError("Cannot set both num_threads and n_jobs")
            self._n_jobs = self.n_jobs
        else:
            if self.num_threads == "auto":
                import multiprocessing

                self._n_jobs = multiprocessing.cpu_count()
            else:
                self._n_jobs = self.num_threads

        if self.n_estimators is not None and self.num_boost_round != 100:
            raise ValueError("Cannot set both n_estimators and num_boost_round")

        self.num_boost_round_ = self.n_estimators or self.num_boost_round

        if y.shape[1] != 1:
            raise ValueError("LightGBMLSS does not support multi-output targets")

        self._y_cols = y.columns
        y_fit = y.iloc[:, 0].to_numpy()
        train_set = lgb.Dataset(X, label=y_fit)

        lgblss_distr = self._get_lgblss_distr(self.dist)
        distr_params = {
            "stabilization": self.stabilization,
            "response_fn": self.response_fn,
            "loss_fn": self.loss_fn,
            "initialize": self.initialize,
        }

        lgblss = _LightGBMLSS(lgblss_distr(**distr_params))

        self.lgb_params_ = {}
        for key in self._lgb_params:
            value = self._n_jobs if key == "num_threads" else getattr(self, key)
            if value is not None:
                self.lgb_params_[key] = value

        if self.n_trials == 0:
            n_rounds = self.num_boost_round_
            train_kwargs = {}

            if self.callbacks is not None:
                train_kwargs["callbacks"] = self.callbacks
        else:
            lgb_params_ = self._hyper_opt(lgblss, train_set)
            n_rounds = lgb_params_.pop("opt_rounds", self.num_boost_round_)
            train_kwargs = {}

            if self.callbacks is not None:
                warnings.warn(
                    "Warning in LightGBMLSS: "
                    "Callbacks are not supported in hyperparameter optimization. "
                    "Ignoring callbacks.",
                    UserWarning,
                    stacklevel=2,
                )

            self.lgb_params_.update(lgb_params_)

        lgblss.train(self.lgb_params_, train_set, num_boost_round=n_rounds, **train_kwargs)

        self.lgblss_ = lgblss
        return self

    def _hyper_opt(self, lgblss, train_set):
        """Run internal hyperparameter optimization."""
        param_dict = {
            "learning_rate": ["float", {"low": 1e-3, "high": 0.3, "log": True}],
            "num_leaves": ["int", {"low": 2, "high": 128, "log": True}],
            "max_depth": ["int", {"low": 1, "high": 12, "log": False}],
            "min_data_in_leaf": ["int", {"low": 1, "high": 128, "log": True}],
            "feature_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
            "bagging_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
            "bagging_freq": ["int", {"low": 1, "high": 7, "log": False}],
        }

        return lgblss.hyper_opt(
            param_dict,
            train_set,
            num_boost_round=self.num_boost_round_,
            nfold=self.nfold,
            early_stopping_rounds=self.early_stopping_rounds,
            max_minutes=self.max_minutes,
            n_trials=self.n_trials,
            silence=True,
            seed=123,
            hp_seed=123,
        ).copy()

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features."""
        index = X.index
        columns = self._y_cols

        y_pred_lgblss = self.lgblss_.predict(X, pred_type="parameters")

        skpro_distr = self._get_skpro_distr(self.dist)
        skpro_distr_vals = self._get_skpro_val_dict(self.dist, y_pred_lgblss)

        return skpro_distr(**skpro_distr_vals, index=index, columns=columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params0 = {"n_trials": 0, "num_boost_round": 5, "num_leaves": 7, "n_jobs": 1}
        params1 = {"dist": "Gamma", "n_trials": 0, "num_boost_round": 5, "n_jobs": 1}
        params2 = {
            "dist": "Weibull",
            "n_trials": 0,
            "num_boost_round": 5,
            "n_jobs": 1,
        }
        params3 = {
            "dist": "TDistribution",
            "n_trials": 0,
            "num_boost_round": 5,
            "n_jobs": 1,
        }
        params4 = {
            "dist": "Laplace",
            "n_trials": 0,
            "num_boost_round": 5,
            "n_jobs": 1,
        }
        params5 = {
            "dist": "Logistic",
            "n_trials": 0,
            "num_boost_round": 5,
            "n_jobs": 1,
        }
        params6 = {"n_trials": 1, "max_minutes": 1, "num_boost_round": 5, "n_jobs": 1}
        params7 = {"dist": "Beta", "n_trials": 0, "num_boost_round": 5, "n_jobs": 1}
        params8 = {
            "n_trials": 0,
            "num_boost_round": 5,
            "learning_rate": 0.1,
            "num_leaves": 5,
            "n_jobs": 1,
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
