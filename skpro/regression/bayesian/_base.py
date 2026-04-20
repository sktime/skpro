# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.regression.base import BaseProbaRegressor


class BaseBayesianRegressor(BaseProbaRegressor):
    """
    Base class for Bayesian probabilistic regressors.

    The class defines a backend-agnostic interface based on posterior
    fitting/prediction hooks. By default, it provides a PyMC + MCMC
    implementation, but subclasses can override the posterior hooks to support
    non-MC Bayesian paradigms (e.g., conjugate/closed-form, variational,
    Laplace, deterministic approximations).

    Subclasses using the default PyMC/MCMC workflow should implement
    `_build_model`. Subclasses using non-PyMC or non-MCMC workflows should
    override `_fit_posterior` and `_predict_proba_from_posterior`.

    Parameters
    ----------
    draws : int, default=1000
        Number of MCMC draws per chain.
    tune : int, default=1000
        Number of tuning steps for MCMC sampler.
    chains : int, default=2
        Number of MCMC chains.
    target_accept : float, default=0.95
        Target acceptance rate for MCMC sampler.
    random_seed : int, optional
        Random seed for MCMC sampling.
    progressbar : bool, default=True
        Whether to show progress bar during sampling.

    Attributes
    ----------
    model_ : pymc.Model
        The fitted PyMC model, if using the default PyMC backend.
    trace_ : arviz.InferenceData
        The posterior samples, if using the default PyMC backend.

    Notes
    -----
    For the default backend, subclasses must implement
    `_build_model(self, X, y)` to construct and return a PyMC model.

    References
    ----------
    - Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
    - Gelman et al. (2013). Bayesian Data Analysis (3rd ed.).
    - Capretto et al. (2022). Bambi: A Bayesian model-building interface in Python JOSS.
    - Polson et al. (2026). Synthetic Priors. arXiv.
    - Xie et al. (2026). Flexible Empirical Bayes for GLMs. arXiv:2601.21217.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["arnavk23"],
        "maintainers": ["arnavk23"],
        "python_version": ">=3.10",
        "python_dependencies": ["pymc"],  # only core dependency
        # estimator tags
        # --------------
        "capability:multioutput": False,  # can the estimator handle multi-output data?
        "capability:missing": True,  # can the estimator handle missing data?
        "X_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit, _predict
        "y_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit
        # CI and test flags
        # -----------------
        "tests:python_dependencies": ["arviz>=0.18.0", "pymc-extras"],
    }

    def update(self, X, y, C=None):
        """Update regressor with new batch of training data (Bayesian updating).

        Parameters
        ----------
        X : pandas DataFrame
            New feature instances.
        y : pandas DataFrame
            New labels.
        C : optional
            Censoring information (not used by default).

        Returns
        -------
        self : reference to self
        """
        # Stub: subclasses should implement Bayesian updating logic
        raise NotImplementedError(
            "Bayesian updating not implemented in base. Subclasses should override."
        )

    def __init__(
        self,
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.95,
        random_seed=None,
        progressbar=True,
        sample_kwargs=None,
        prior_config=None,
        inference_strategy="mcmc",
        prior_strength=None,
        hyperprior_shape=None,
        robust=False,
        **kwargs,
    ):
        """Initialize a Bayesian probabilistic regressor.

        Parameters
        ----------
        prior_config : dict or None, default=None
            Dictionary specifying prior distributions or hyperparameters for model
            parameters. Example: {"beta": {"dist": "Normal", "mean": 0, "sd": 10},
            "sigma": {"dist": "HalfNormal", "sd": 5}}
            Subclasses can read this in _build_model (PyMC path) or _fit_posterior
            (custom path). If None, subclasses fall back to weakly informative defaults.
        inference_strategy : str, default="mcmc"
            Inference method. Options: "mcmc", "conjugate", "variational".
        robust : bool, default=False
            If True, use heavier-tailed priors (e.g., Student-t) for location/scale.
        prior_strength : float, optional
            Strength/weight of the prior (e.g., for synthetic/power priors).
        hyperprior_shape : float, optional
            Shape parameter for hierarchical/empirical Bayes priors.
        draws, tune, chains, target_accept, random_seed, progressbar, sample_kwargs :
            MCMC sampling parameters (see class docstring).
        kwargs : dict
            Additional keyword arguments passed to BaseProbaRegressor.
        """
        self.prior_config = prior_config if prior_config is not None else {}
        self.inference_strategy = inference_strategy
        self.robust = robust
        self.prior_strength = prior_strength
        self.hyperprior_shape = hyperprior_shape
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.progressbar = progressbar
        self.sample_kwargs = {} if sample_kwargs is None else sample_kwargs

        super().__init__(**kwargs)

    def _get_default_priors(self, X, y):
        """Return default weakly informative priors (can be overridden).

        Returns
        -------
        dict
            Dict compatible with prior_config merging.
        """
        import numpy as np

        n_features = X.shape[1]
        y_std = y.values.std() if hasattr(y, "values") else y.std()
        sd_intercept = 10.0 * y_std
        sd_slopes = 2.5 / np.sqrt(n_features)
        sd_noise = y_std
        defaults = {
            "intercept": {"dist": "Normal", "mean": 0.0, "sd": sd_intercept},
            "slopes": {"dist": "Normal", "mean": 0.0, "sd": sd_slopes},
            "noise": {"dist": "HalfNormal", "sd": sd_noise},
        }
        if getattr(self, "robust", False):
            defaults["intercept"]["dist"] = "StudentT"
            defaults["intercept"]["nu"] = 4
            defaults["slopes"]["dist"] = "StudentT"
            defaults["slopes"]["nu"] = 4
        if getattr(self, "prior_strength", None) is not None:
            strength = self.prior_strength
            defaults["intercept"]["sd"] /= strength**0.5
            defaults["slopes"]["sd"] /= strength**0.5
            defaults["noise"]["sd"] /= strength**0.5
        return defaults

    def _apply_prior_config(self, model_vars, prior_cfg):
        """Apply user prior_config to PyMC variables or subclass logic."""
        import re

        for var_name, spec in prior_cfg.items():
            if isinstance(spec, str):
                # Parse e.g. "Normal(0,10)" to dict
                m = re.match(r"([A-Za-z]+)\(([^)]+)\)", spec)
                if m:
                    dist = m.group(1)
                    params = [float(x) for x in m.group(2).split(",")]
                    if dist.lower() == "normal":
                        spec = {"dist": "Normal", "mean": params[0], "sd": params[1]}
                    elif dist.lower() == "studentt":
                        spec = {
                            "dist": "StudentT",
                            "mean": params[0],
                            "sd": params[1],
                            "nu": params[2],
                        }
                    # Add more as needed
            if var_name in model_vars:
                # Override prior (PyMC or custom logic)
                model_vars[var_name].set_prior(spec)
        return model_vars

    # Optionally, utility for prior parsing
    def _parse_prior(self):
        """Parse prior specification into usable form for inference."""
        # Example: convert string to dict, validate shapes, etc.
        return self.prior

    def _fit(self, X, y):
        """Fit regressor to training data via posterior inference backend."""
        self._fit_posterior(X, y)
        return self

    def _fit_posterior(self, X, y):
        """Fit regressor to training data.

        Parameters
        ----------
        X : pandas DataFrame
            Feature instances to fit regressor to.
        y : pandas DataFrame
            Labels to fit regressor to.

        Returns
        -------
        self : BaseBayesianRegressor
            Fitted regressor.
        """
        if self.inference_strategy == "mcmc":
            import warnings

            import pandas as pd
            import pymc as pm

            # Merge default priors with user config
            prior_cfg = {**self._get_default_priors(X, y), **self.prior_config}
            # Build the PyMC model using subclass implementation
            self.model_ = self._build_model(X, y, prior_cfg)

            # Perform MCMC sampling
            with self.model_:
                self.trace_ = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    target_accept=self.target_accept,
                    random_seed=self.random_seed,
                    progressbar=self.progressbar,
                    return_inferencedata=True,
                    **self.sample_kwargs,
                )

            # Add training data to trace
            training_data = pd.concat([X, y], axis=1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.trace_.add_groups(training_data=training_data.to_xarray())
        elif self.inference_strategy == "variational":
            self._fit_variational_posterior(X, y)
        elif self.inference_strategy == "conjugate":
            # Subclass should implement conjugate closed-form
            raise NotImplementedError("Conjugate inference not implemented in base.")
        else:
            raise ValueError(f"Unknown inference_strategy: {self.inference_strategy}")

    def _fit_variational_posterior(self, X, y):
        """Fit mean-field variational posterior (ADVI)."""
        import pymc as pm

        prior_cfg = {
            **self._get_default_priors(X, y),
            **self.prior_config,
        }
        self.model_ = self._build_model(X, y, prior_cfg)
        with self.model_:
            self.approx_ = pm.fit(
                method="advi",
                n=30000,
                obj_optimizer=pm.adagrad_window(learning_rate=0.01),
                callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)],
                progressbar=self.progressbar,
            )
            self.trace_ = self.approx_.sample(self.draws)

    def _build_model(self, X, y, prior_cfg=None):
        """Build the PyMC model.

        This method is used by the default PyMC/MCMC backend and must be
        implemented by subclasses that rely on that backend.

        Parameters
        ----------
        X : pandas DataFrame
            Feature instances.
        y : pandas DataFrame
            Labels.
        prior_cfg : dict, optional
            Prior configuration dictionary.

        Returns
        -------
        model : pymc.Model
            The constructed PyMC model.
        """
        raise NotImplementedError("Subclasses must implement the _build_model method.")

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features."""
        return self._predict_proba_from_posterior(X)

    def _predict_proba_from_posterior(self, X):
        """Predict distribution over labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            Data to predict labels for.

        Returns
        -------
        pred_dist : skpro BaseDistribution
            Predicted distributions.
        """
        import pymc as pm

        from skpro.distributions import Empirical

        with self.model_:
            # Remove existing predictions if any
            if "predictions" in self.trace_.groups():
                del self.trace_.predictions

            # Set data for prediction
            self._set_prediction_data(X)

            # Sample posterior predictive
            self.trace_.extend(
                pm.sample_posterior_predictive(
                    self.trace_,
                    predictions=True,
                    random_seed=self.random_seed,
                )
            )

            # Mark that prediction has been done
            self._predict_done = True

        pred_var_name = self._get_predictive_variable_name()
        pred_xarray = self.trace_.predictions[pred_var_name]
        pred_df = pred_xarray.to_dataframe().reset_index()

        if {"chain", "draw"}.issubset(pred_df.columns):
            pred_df["sample_id"] = pred_df.groupby(
                ["chain", "draw"], sort=False
            ).ngroup()
            sample_dim_cols = ["chain", "draw"]
        elif "sample" in pred_df.columns:
            pred_df["sample_id"] = pred_df["sample"]
            sample_dim_cols = ["sample"]
        else:
            raise ValueError(
                "Posterior predictive samples must expose either ('chain', 'draw') "
                "or 'sample' coordinates to build an Empirical distribution."
            )

        obs_dims = [
            dim
            for dim in pred_xarray.dims
            if dim not in sample_dim_cols and dim not in ["chain", "draw"]
        ]
        if len(obs_dims) != 1:
            raise NotImplementedError(
                "Default posterior predictive conversion expects a single observation "
                "dimension. Override `_predict_proba_from_posterior` for custom shapes."
            )

        obs_dim = obs_dims[0]
        target_col = getattr(self, "_y_columns", ["y"])[0]
        pred_df = pred_df[[obs_dim, "sample_id", pred_var_name]]
        pred_df = pred_df.rename(columns={obs_dim: "obs_id", pred_var_name: target_col})
        pred_df = pred_df.set_index(["sample_id", "obs_id"])

        # Create Empirical distribution
        pred_dist = Empirical(spl=pred_df, columns=[target_col], index=X.index)

        return pred_dist

    def _set_prediction_data(self, X):
        """Set data for prediction in the PyMC model.

        This method can be overridden by subclasses if they need
        custom data setting logic.

        Parameters
        ----------
        X : pandas DataFrame
            Prediction features.
        """
        import pymc as pm

        # Default: assume X is set as "X" in the model
        pm.set_data({"X": X}, coords={"obs_id": X.index, "pred_id": X.columns})

    def _get_predictive_variable_name(self):
        """Get predictive variable name from trace predictions group."""
        if not hasattr(self, "trace_") or "predictions" not in self.trace_.groups():
            raise ValueError("No 'predictions' group found in posterior trace.")

        predictions = self.trace_.predictions
        configured = getattr(self, "_predictive_var_name", None)
        if configured is not None:
            if configured not in predictions:
                raise ValueError(
                    f"Configured predictive variable '{configured}' not found in "
                    f"predictions group: {list(predictions.data_vars)}"
                )
            return configured

        if "y_obs" in predictions:
            return "y_obs"

        data_vars = list(predictions.data_vars)
        if len(data_vars) == 1:
            return data_vars[0]

        raise ValueError(
            "Could not infer predictive variable from predictions group. "
            "Set `self._predictive_var_name` in subclass or override "
            "`_predict_proba_from_posterior`."
        )

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
            Dictionary containing fitted parameters, including:
            - trace_: arviz.InferenceData with MCMC results
        """
        return self._get_fitted_params_from_posterior()

    def _get_fitted_params_from_posterior(self):
        """Get fitted parameters from the posterior representation."""
        fitted_params = {}
        if hasattr(self, "trace_"):
            fitted_params["trace_"] = self.trace_
        return fitted_params

    def get_posterior_summary(self, **kwargs):
        """Get summary statistics of the posterior distributions."""
        return self._get_posterior_summary_from_posterior(**kwargs)

    def _get_posterior_summary_from_posterior(self, **kwargs):
        """Get summary statistics of the posterior distributions."""
        if hasattr(self, "approx_"):
            import arviz as az

            return az.summary(
                self.approx_.sample(self.draws), kind="stats", extend=True
            )
        elif hasattr(self, "trace_"):
            import arviz as az

            return az.summary(self.trace_, kind="diagnostics", extend=True, **kwargs)
        else:
            raise NotImplementedError("No posterior available for summary.")

    inference_strategies_supported = ["mcmc", "variational"]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances.
        """
        params1 = {}
        params2 = {
            "draws": 100,
            "tune": 100,
            "chains": 1,
            "random_seed": 42,
        }
        params3 = {
            "inference_strategy": "variational",
            "draws": 50,
            "robust": True,
        }
        params4 = {
            "prior_config": {"intercept": "Normal(0,10)", "slopes": "StudentT(0,5,4)"},
            "prior_strength": 2.0,
        }
        return [params1, params2, params3, params4]
