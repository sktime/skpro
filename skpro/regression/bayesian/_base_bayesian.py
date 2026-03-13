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
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["skpro developers"],
        "python_version": ">=3.10",
        # estimator tags
        # --------------
        "capability:multioutput": False,
        "capability:missing": True,
        "capability:update": False,  # Bayesian updating not implemented yet
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(
        self,
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.95,
        random_seed=None,
        progressbar=True,
        sample_kwargs=None,
    ):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.progressbar = progressbar
        self.sample_kwargs = {} if sample_kwargs is None else sample_kwargs

        super().__init__()

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
        import warnings

        import pandas as pd
        import pymc as pm

        # Build the PyMC model using subclass implementation
        self.model_ = self._build_model(X, y)

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

    def _build_model(self, X, y):
        """Build the PyMC model.

        This method is used by the default PyMC/MCMC backend and must be
        implemented by subclasses that rely on that backend.

        Parameters
        ----------
        X : pandas DataFrame
            Feature instances.
        y : pandas DataFrame
            Labels.

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
        try:
            pm.set_data({"X": X}, coords={"obs_id": X.index, "pred_id": X.columns})
        except Exception as exc:
            raise RuntimeError(
                "Default `_set_prediction_data` expects a PyMC model with mutable "
                "data variable 'X' and coordinates 'obs_id'/'pred_id'. "
                "Override `_set_prediction_data` for custom models."
            ) from exc

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
        """Get summary statistics of the posterior distributions.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to arviz.summary.

        Returns
        -------
        summary : pandas.DataFrame
            Summary statistics of posterior distributions.
        """
        import arviz as az

        if not hasattr(self, "trace_"):
            raise NotImplementedError(
                "Posterior summary is not available by default for this backend. "
                "Override `_get_posterior_summary_from_posterior` in subclasses "
                "that do not use `trace_`/ArviZ."
            )

        return az.summary(self.trace_, **kwargs)

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

        return [params1, params2]
