# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.regression.base import BaseProbaRegressor


class BaseBayesianRegressor(BaseProbaRegressor):
    """
    Base class for Bayesian probabilistic regressors using PyMC.

    This class encapsulates the PyMC backend logic for MCMC sampling and
    posterior predictive inference. Individual Bayesian estimators should
    inherit from this class and implement the `_build_model` method to define
    their specific probabilistic model structure.

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
        The fitted PyMC model.
    trace_ : arviz.InferenceData
        The MCMC sampling results.

    Notes
    -----
    Subclasses must implement the `_build_model(self, X, y)` method that
    constructs and returns a PyMC model given the training data X and y.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["skpro developers"],
        "python_version": ">=3.10",
        "python_dependencies": [
            "pymc>=5.0.0",
            "arviz>=0.18.0",
        ],
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
    ):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.progressbar = progressbar

        super().__init__()

    def _fit(self, X, y):
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
            )

        # Add training data to trace
        training_data = pd.concat([X, y], axis=1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.trace_.add_groups(training_data=training_data.to_xarray())

        return self

    def _build_model(self, X, y):
        """Build the PyMC model.

        This method must be implemented by subclasses to define the
        specific probabilistic model structure.

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

        # Convert to skpro Empirical distribution
        # Assume y_obs is the predictive variable
        if "y_obs" not in self.trace_.predictions:
            raise ValueError("Model must have 'y_obs' variable for predictions.")

        pred_xarray = self.trace_.predictions["y_obs"]

        # Convert to DataFrame format expected by Empirical
        pred_df = pred_xarray.to_dataframe().reset_index()

        # Create sample_id by combining chain and draw
        pred_df["sample_id"] = pred_df["chain"] * self.draws + pred_df["draw"]

        # Format for Empirical: columns should be the target variable names
        # Assume single output for now
        target_col = getattr(self, "_y_columns", ["y"])[0]
        pred_df = pred_df[["obs_id", "sample_id", "y_obs"]]
        pred_df = pred_df.rename(columns={"y_obs": target_col})
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

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
            Dictionary containing fitted parameters, including:
            - trace_: arviz.InferenceData with MCMC results
        """
        return {"trace_": self.trace_}

    def get_posterior_summary(self, **kwargs):
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
