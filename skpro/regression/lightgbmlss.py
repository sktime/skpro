"""Interface for lightgbmlss probabilistic regressor."""
from skpro.regression.adapters._statmixedml import _StatMixedMLMixin


class LightGBMLSS(_StatMixedMLMixin):
    """Interface to lightgbmlss regressor from the lightgbmlss package.

    Direct interface to ``LightGBMLSS`` from ``lightgbmlss`` package by ``StatMixedML``.

    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["yuvimittal"],
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

        opt_params.setdefault("n_jobs", self._n_cpu)

        # Train Model with optimized hyperparameters
        lgblss.train(opt_params, dtrain, num_boost_round=n_rounds)

        self.lgblss_ = lgblss
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
        index = X.index
        y_cols = self._y_cols
        columns = y_cols

        # lightgbmlss.predict takes raw feature matrix / DataFrame
        y_pred_lgblss = self.lgblss_.predict(X, pred_type="parameters")

        skpro_distr = self._get_skpro_distr(self.dist)
        skpro_distr_vals = self._get_skpro_val_dict(self.dist, y_pred_lgblss)

        y_pred = skpro_distr(**skpro_distr_vals, index=index, columns=columns)

        return y_pred
