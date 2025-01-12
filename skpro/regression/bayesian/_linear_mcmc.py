"""
Simple Bayesian Linear Regressor.

Bayesian Linear Regression defined with user-specified priors or defaults for slopes,
intercept, and noise; implemented using the pymc backend.
"""

# copyright: skpro developers
__author__ = ["meraldoantonio"]

from skbase.utils.dependencies import _check_soft_dependencies

from skpro.regression.base import BaseProbaRegressor


class BayesianLinearRegressor(BaseProbaRegressor):
    """
    Bayesian Linear Regression class with MCMC sampling.

    Defined with user-specified priors or defaults for slopes, intercept,
    and noise; implemented using the pymc backend.

    Parameters
    ----------
    prior_config : Dictionary, optional
        Dictionary of priors
        Class-default defined by default_prior_config method.
    sampler_config : Dictionary, optional
        Dictionary of parameters that initialise sampler configuration.
        Class-default defined by default_sampler_config method.

    Example
    -------
    >>> from skpro.regression.bayesian import BayesianLinearRegressor
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>> from sklearn.model_selection import train_test_split  # doctest: +SKIP
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)  # doctest: +SKIP

    >>> bayes_model = BayesianLinearRegressor()  # doctest: +SKIP
    >>> bayes_model.fit(X_train, y_train)  # doctest: +SKIP
    >>> y_test_pred_proba = bayes_model.predict_proba(X_test)  # doctest: +SKIP
    >>> y_test_pred = bayes_model.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["meraldoantonio"],
        "python_version": ">=3.10",
        "python_dependencies": [
            "pymc",
            "pymc_marketing",
            "arviz>=0.18.0",
        ],
        # estimator tags
        # --------------
        "capability:multioutput": False,  # can the estimator handle multi-output data?
        "capability:missing": True,  # can the estimator handle missing data?
        "X_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit, _predict
        "y_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit
    }

    def __init__(self, prior_config=None, sampler_config=None):
        if sampler_config is None:
            sampler_config = {}
        if prior_config is None:
            prior_config = {}  # configuration for priors
        self.sampler_config = {**self.default_sampler_config, **sampler_config}
        self.prior_config = {**self.default_prior_config, **prior_config}
        self.model = None  # generated during fitting
        self.idata = None  # generated during fitting
        self._predict_done = False  # a flag indicating if a prediction has been done

        print(  # noqa: T201
            f"instantiated {self.__class__.__name__} with the following priors:"
        )

        for key, value in self.prior_config.items():
            print(f"  - {key}: {value}")  # noqa: T201

        super().__init__()

    @property
    def default_prior_config(self):
        """Return a dictionary of prior defaults."""
        from pymc_marketing.prior import Prior

        print(  # noqa: T201
            "The model assumes that the intercept and slopes are independent. \n\
            Modify the model if this assumption doesn't apply!"
        )
        default_prior_config = {
            "intercept": Prior(
                "Normal", mu=0, sigma=100
            ),  # Weakly informative normal prior with large sigma
            "slopes": Prior(
                "Normal", mu=0, sigma=100, dims=("pred_id",)
            ),  # Same for slopes
            "noise_var": Prior(
                "HalfCauchy", beta=5
            ),  # Weakly informative Half-Cauchy prior for noise variance
        }
        return default_prior_config

    @property
    def default_sampler_config(self):
        """Return a class default sampler configuration dictionary."""
        default_sampler_config = {
            "draws": 1000,
            "tune": 1000,
            "chains": 2,
            "target_accept": 0.95,
            "random_seed": 123,
            "progressbar": True,
        }
        return default_sampler_config

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
        import warnings

        import pandas as pd
        import pymc as pm

        assert len(y.columns) == 1, "y must have only one column!"
        self._X = X
        self._y = y
        self._y_vals = y.values[
            :, 0
        ]  # we need a 1-dimensional array for compatibility with pymc

        # Model construction and posterior sampling
        with pm.Model(coords={"obs_id": X.index, "pred_id": X.columns}) as self.model:
            # Mutable data containers for X and y
            X_data = pm.Data("X", X, dims=("obs_id", "pred_id"))
            y_data = pm.Data("y", self._y_vals, dims=("obs_id"))

            # Priors for model parameters, taken from self.prior_config
            self.intercept = self.prior_config["intercept"].create_variable("intercept")
            self.slopes = self.prior_config["slopes"].create_variable("slopes")
            self.noise_var = self.prior_config["noise_var"].create_variable("noise_var")
            self.noise = pm.Deterministic("noise", self.noise_var**0.5)

            # Expected value of the target variable
            self.mu = pm.Deterministic(
                "mu", self.intercept + pm.math.dot(X_data, self.slopes)
            )

            # Likelihood of observations
            y_obs = pm.Normal(  # noqa: F841
                "y_obs", mu=self.mu, sigma=self.noise, observed=y_data, dims=("obs_id")
            )

            # Constructing the posterior
            self.idata = pm.sample(**self.sampler_config)

        # Incorporation of training_data as a new group in self.idata
        training_data = pd.concat([X, y], axis=1)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
            )
            self.idata.add_groups(training_data=training_data.to_xarray())
        return self

    def visualize_model(self, **kwargs):
        """Use Graphviz to visualize the model flow."""
        _check_soft_dependencies(
            "graphviz", msg="You need to install Graphviz to use this method!"
        )
        import pymc as pm

        assert self._is_fitted, "You need to fit the model before visualizing it!"

        return pm.model_to_graphviz(self.model, **kwargs)

    def _sample_dataset(self, group_name, return_type=None):
        """
        General method to sample from a specified group in the idata object.

        Extracts samples from a specified group (e.g., 'prior') in the idata object and
        returns them in the required format

        Parameters
        ----------
        group_name : str
            The name of the group in the idata object to sample from (e.g., 'prior').

        return_type : str or None, optional (default=None)
            The format in which to return the sampled distributions.
            Accepted values are:
            - "xarray": Returns an xarray.Dataset
            - "numpy": Returns a dictionary of NumPy arrays
            - "dataframe": Returns a pandas DataFrame
            - "skpro": Returns an `Empirical` distribution from the skpro library.
            - None: Does not return any sampled data but performs the sampling
                    and updates the 'idata' attribute.

        Returns
        -------
        xarray.Dataset or dict or pd.DataFrame or skpro.distributions.Empirical or None

            The sampled distributions in the specified format,
            or None if return_type is None.
        """
        import pandas as pd

        # Validate the return_type
        assert return_type in [
            "xarray",
            "numpy",
            "dataframe",
            "skpro",
            None,
        ], "return_type must be one of 'xarray', 'numpy', 'dataframe', 'skpro', or None"

        # Validate that the group_name exists in idata
        assert hasattr(
            self.idata, group_name
        ), f"{group_name} group does not exist in the idata object."

        # Get the specified group from idata
        group = getattr(self.idata, group_name)
        # prediction-specific groups which focus on posterior predictive
        is_predictive = group_name in ["predictions", "posterior_predictive"]
        # as opposed to ["prior", "posterior"] which focus on prior/posterior

        if is_predictive:
            variables = ["y_obs"]
        else:
            variables = ["intercept", "slopes", "noise_var", "noise"]

        if return_type is None:
            return None
        elif return_type == "xarray":
            return group
        else:
            data_dict = {}

            for var in variables:
                # Check if the variable has a `pred_id` dimension
                if var in group and "pred_id" in group[var].dims:
                    # Iterate through each feature (e.g., 'feature1', 'feature2')
                    for feature in group[var].pred_id.values:
                        # Select the slope for the current feature and flatten it
                        feature_key = f"{var}_{feature}"
                        data_dict[feature_key] = (
                            group[var]
                            .sel(pred_id=feature)
                            .stack({"sample": ("chain", "draw")})
                            .values.squeeze()
                        )
                else:
                    if var in group:
                        data_dict[var] = (
                            group[var]
                            .stack({"sample": ("chain", "draw")})
                            .values.squeeze()
                        )

            if return_type == "numpy":
                return data_dict

            elif return_type == "dataframe":
                if is_predictive:
                    return pd.DataFrame(data_dict["y_obs"]).T
                else:
                    return pd.DataFrame(data_dict)

            elif return_type == "skpro":
                from skpro.distributions import Empirical

                if not is_predictive:
                    df = pd.DataFrame(data_dict)
                    reshaped_df = df.stack()
                    reshaped_df = reshaped_df.reset_index(name="value")
                    reshaped_df.set_index(["level_0", "level_1"], inplace=True)
                    reshaped_df.index.names = ["obs_id", "variable"]
                    return Empirical(spl=reshaped_df)
                else:
                    # Extract posterior predictive distributions as an xarray DataArray
                    pred_proba_xarray = group["y_obs"]

                    # Convert data to pd.DataFrame and format it appropriately for
                    # subsequent conversion into a skpro Empirical distribution
                    pred_proba_df = pred_proba_xarray.to_dataframe()
                    pred_proba_df = pred_proba_df.reset_index()

                    # Create a new 'sample_id' column by
                    # combining the 'chain' and 'draw' columns
                    pred_proba_df["sample_id"] = (
                        pred_proba_df["chain"] * self.sampler_config["draws"]
                        + pred_proba_df["draw"]
                    )
                    pred_proba_df = pred_proba_df[["obs_id", "sample_id", "y_obs"]]
                    pred_proba_df = pred_proba_df.rename(
                        columns={"y_obs": self._y.columns[0]}
                    )
                    pred_proba_df = pred_proba_df.set_index(["sample_id", "obs_id"])

                    # Convert data to skpro Empirical distribution
                    pred_proba_dist = Empirical(
                        spl=pred_proba_df, columns=self._y.columns
                    )
                    return pred_proba_dist

    def _get_dataset_summary(self, group_name, var_names=None, **kwargs):
        """
        Get the summary statistics of a specified group in the idata object.

        Parameters
        ----------
        group_name : str
            The name of the group in the idata object to summarize (e.g., 'prior').

        var_names : list, optional (default=None)
            A list of variable names to include in the summary.
            If None, all variables in the group are included.

        **kwargs :
            Additional keyword arguments to pass to `arviz.summary`.

        Returns
        -------
        az.data.inference_data.Summary
            The summary statistics for the specified group and variables.
        """
        import arviz as az

        # Check if the specified group exists in the idata object
        if group_name not in self.idata.groups():
            if group_name == "prior":
                self.sample_prior()
            elif group_name == "posterior":
                self.sample_posterior()
            else:
                raise ValueError(
                    f"Group '{group_name}' does not exist in the idata object."
                )

        # Get the summary statistics with optional kwargs
        return az.summary(
            getattr(self.idata, group_name), var_names=var_names, **kwargs
        )

    def sample_prior(self, return_type=None):
        """
        Sample from the prior distributions.

        Samples from the prior distributions and returns
        them in the required format

        If return_type is None, the method updates the 'idata' attribute
        by adding the 'prior' group but does not return any samples.

        return_type : str or None, optional (default=None)
            The format in which to return the sampled distributions.
            Accepted values are:
            - "xarray": Returns an xarray.Dataset
            - "numpy": Returns a dictionary of NumPy arrays
            - "dataframe": Returns a pandas DataFrame
            - "skpro": Returns an `Empirical` distribution from the skpro library.
            - None: Does not return any sampled data but performs the sampling
                    and updates the 'idata' attribute.

        Returns
        -------
        xarray.Dataset or dict or pd.DataFrame or skpro.distributions.Empirical or None
            The sampled distributions in the specified format,
            or None if return_type is None.
        """
        import pymc as pm

        assert (
            self.is_fitted
        ), "Model needs to be fitted before you can sample from prior"

        with self.model:
            # if we've previously used the model for prediction,
            # we need to reset the reference of 'X' to X used for training
            if self._predict_done:
                pm.set_data(
                    {"X": self._X},
                    coords={"obs_id": self._X.index, "pred_id": self._X.columns},
                )
            self.idata.extend(
                pm.sample_prior_predictive(
                    samples=self.sampler_config["draws"],
                    random_seed=self.sampler_config["random_seed"],
                )
            )  # todo: the keyword 'samples' will be changed to 'draws'
            # in pymc 5.16

        return self._sample_dataset(
            group_name="prior",
            return_type=return_type,
        )

    def get_prior_summary(self, **kwargs):
        """
        Get the summary statistics of prior distributions.

        Parameters
        ----------
        **kwargs :
            Additional keyword arguments to pass to `arviz.summary`.

        Returns
        -------
        az.data.inference_data.Summary
            The summary statistics for the prior distributions.
        """
        return self._get_dataset_summary(
            group_name="prior",
            var_names=["intercept", "slopes", "noise_var", "noise"],
            **kwargs,
        )

    def sample_posterior(self, return_type=None):
        """
        Sample from the posterior distributions.

        Samples from the posterior distributions and returns
        them in the required format

        If return_type is None, the method updates the 'idata' attribute
        by adding the 'posterior' group but does not return any samples.

        return_type : str or None, optional (default="xarray")
            The format in which to return the sampled distributions.
            Accepted values are:
            - "xarray": Returns an xarray.Dataset
            - "numpy": Returns a dictionary of NumPy arrays
            - "dataframe": Returns a pandas DataFrame
            - "skpro": Returns an `Empirical` distribution from the skpro library.
            - None: Does not return any sampled data but performs the sampling
                    and updates the 'idata' attribute.

        Returns
        -------
        xarray.Dataset or dict or pd.DataFrame or skpro.distributions.Empirical or None
            The sampled distributions in the specified format,
            or None if return_type is None.
        """
        assert (
            self.is_fitted
        ), "The model must be fitted before posterior can be returned."
        return self._sample_dataset(
            group_name="posterior",
            return_type=return_type,
        )

    def get_posterior_summary(self, **kwargs):
        """
        Get the summary statistics of the posterior distributions.

        Parameters
        ----------
        **kwargs :
            Additional keyword arguments to pass to `arviz.summary`.

        Returns
        -------
        az.data.inference_data.Summary
            The summary statistics for the posterior distributions.
        """
        return self._get_dataset_summary(
            group_name="posterior",
            var_names=["intercept", "slopes", "noise_var", "noise"],
            **kwargs,
        )

    def sample_in_sample_posterior_predictive(self, return_type=None):
        """Perform in-sample predictions and sample from it."""
        import pymc as pm

        with self.model:
            # if we've previously used the model for prediction,
            # we need to reset the reference of 'X' to X_train (i.e. self._X)
            if self._predict_done:
                pm.set_data(
                    {"X": self._X},
                    coords={"obs_id": self._X.index, "pred_id": self._X.columns},
                )
            self.idata.extend(
                pm.sample_posterior_predictive(self.idata, predictions=False)
            )

        return self._sample_dataset(
            group_name="posterior_predictive", return_type=return_type
        )

    def plot_ppc(self, **kwargs):
        """Plot the posterior predictive check."""
        import arviz as az

        if "posterior_predictive" not in self.idata:
            self.sample_in_sample_posterior_predictive()

        return az.plot_ppc(self.idata, **kwargs)

    def _predict_proba(self, X):
        """
        Predict distribution over labels for data from features.

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
        pred_proba_dist : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        import pymc as pm

        with self.model:
            if "predictions" in self.idata.groups():
                del self.idata.predictions

            # Set the X to be the new 'X' variable and then sample posterior predictive
            pm.set_data({"X": X}, coords={"obs_id": X.index, "pred_id": X.columns})
            self.idata.extend(
                pm.sample_posterior_predictive(
                    self.idata,
                    predictions=True,
                )
            )
            self._predict_done = True  # a flag indicating prediction has been done

        return self._sample_dataset(group_name="predictions", return_type="skpro")

    # todo: return default parameters, so that a test instance can be created
    # required for automated unit and integration testing of estimator
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
        from pymc_marketing.prior import Prior

        params1 = {}
        params2 = {"prior_config": {"intercept": Prior("Normal", mu=0, sigma=10)}}

        return [params1, params2]
