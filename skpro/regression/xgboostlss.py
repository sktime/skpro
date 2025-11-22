"""Interface for xgboostlss probabilistic regressor."""

from skpro.regression.adapters._statmixedml import _StatMixedMLMixin


class XGBoostLSS(_StatMixedMLMixin):
    """Interface to xgboostlss regerssor from the xgboostlss package.

    Direct interface to ``XGBoostLSS`` from ``xgboostlss`` package by ``StatMixedML``.
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                stabilization=self.stabilization,
                response_fn=self.response_fn,
                loss_fn=self.loss_fn,
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
