"""Simple Bayesian Linear Regressor with normal priors for slopes and intercept and half-normal prior for noise; coded on pymc backend"""
# copyright: skpro developers

__author__ = ["meraldoantonio"]

from skpro.regression.base import BaseProbaRegressor

import pandas as pd
import numpy as np

class BayesianLinearRegressor(BaseProbaRegressor):
    """Bayesian Linear Regressor with normal priors for slopes and intercept and half-normal prior for noise.

    Parameters
    ----------
    intercept_mu : float, optional (default=0)
        Mean of the normal prior for the intercept.
    intercept_sigma : float, optional (default=10)
        Standard deviation of the normal prior for the intercept.
    slopes_mu : float, optional (default=0)
        Mean of the normal prior for the slopes.
    slopes_sigma : float, optional (default=10)
        Standard deviation of the normal prior for the slopes.
    noise_sigma : float, optional (default=10)
        Standard deviation of the half-normal prior for the noise.
    chains : int, optional (default=2)
        Number of MCMC chains to run.
    draws : int, optional (default=2000)
        Number of MCMC draws to sample from each chain; will be applied to all sampling steps.

    Example
    -------
    >>> from skpro.regression.bayesian import BayesianLinearRegressor
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>> from sklearn.model_selection import train_test_split  # doctest: +SKIP
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)  # doctest: +SKIP

    >>> bayes_model= BayesianLinearRegressor()  # doctest: +SKIP
    >>> bayes_model.fit(X_train, y_train)  # doctest: +SKIP
    >>> y_test_pred_proba = bayes_model.predict_proba(X_test)  # doctest: +SKIP
    >>> y_test_pred = bayes_model.predict(X_test)  # doctest: +SKIP
    """
    _tags = {
        # packaging info
        # --------------
        "authors": ["meraldoantonio"],  # authors, GitHub handles
        "python_version": None,
        "python_dependencies": ["pymc"],

        # estimator tags
        # --------------
        "capability:multioutput": False,  # can the estimator handle multi-output data?
        "capability:missing": True,  # can the estimator handle missing data?
        "X_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit, _predict
        "y_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit
    }


    def __init__(self, intercept_mu=0, intercept_sigma=10, slopes_mu=0, slopes_sigma=10, noise_sigma=10, chains=2, draws=2000):

        # hyperparameters for priors
        self.intercept_sigma = intercept_sigma
        self.intercept_mu = intercept_mu
        self.slopes_sigma = slopes_sigma
        self.slopes_mu = slopes_mu
        self.noise_sigma = noise_sigma
        self.chains = chains
        self.draws = draws

        super().__init__()

        # Assertions to check validity of input parameters
        assert self.intercept_sigma > 0, "intercept_sigma must be positive"
        assert self.slopes_sigma > 0, "slopes_sigma must be positive"
        assert self.noise_sigma > 0, "noise_sigma must be positive"
        assert isinstance(self.chains, int) and self.chains > 0, "chains must be a positive integer"
        assert isinstance(self.draws, int) and self.draws > 0, "draws must be a positive integer"

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

        import pymc as pm
        assert len(y.columns) == 1, "y must have only one column!"
        self._X = X
        self._y = y
        self._y_vals = y.values[:,0] # we need a 1-dimensional array for compatibility with pymc
        self._X_cols = X.columns
        self._y_cols = y.columns
        self._predict_done = False

        with pm.Model(coords={"obs_id": X.index, "pred_id": X.columns}) as self.model:

            # Mutable data containers
            X_data = pm.Data("X", self._X, dims = ("obs_id", "pred_id"))
            y_data = pm.Data("y", self._y_vals, dims = ("obs_id"))

            # Priors for unknown model parameters
            self.intercept = pm.Normal("intercept", mu=self.intercept_mu, sigma=self.intercept_sigma)
            self.slopes = pm.Normal("slopes", mu=self.slopes_mu, sigma=self.slopes_sigma, shape = self._X.shape[1], dims=("pred_id"))
            self.noise = pm.HalfNormal("noise", sigma=self.noise_sigma)

            # Expected value of outcome
            self.mu = pm.Deterministic("mu", self.intercept + pm.math.dot(X_data, self.slopes))

            # Likelihood (sampling distribution) of observations
            y_obs = pm.Normal("y_obs", mu=self.mu, sigma=self.noise, observed=y_data, dims =("obs_id"))

            # Constructing the posterior
            self.trace = pm.sample(chains = self.chains, draws = self.draws)

            # Constructing the in-sample posterior predictive
            self.trace.extend(pm.sample_posterior_predictive(self.trace))
            
        return self

    def get_prior(self, return_type = "xarray"):
        """Extracts the prior distribution"""
        import pymc as pm

        assert return_type in ["xarray", "numpy", "dataframe"], "return_type must be one of 'xarray', 'numpy' or 'dataframe"
        
        with self.model:
            # if we've previously used the model for prediction, we need to reset the reference of 'X' to the training data
            if self._predict_done:
                pm.set_data({"X": self._X}, coords={"obs_id": self._X.index, "pred_id": self._X.columns})
                # todo: uniformize the number of chains during sampling
            self.trace.extend(pm.sample_prior_predictive(samples=self.draws))
            prior = self.trace.prior

        if return_type == "xarray":
            return prior
        elif return_type == "numpy":
            intercept = prior["intercept"].values.squeeze()
            slopes = prior["slopes"].values.squeeze()
            noise = prior["noise"].values.squeeze()
            return {"intercept": intercept, "slopes": slopes, "noise": noise}
        else:
            intercept = prior["intercept"].values.squeeze()
            slopes = prior["slopes"].values.squeeze()
            noise = prior["noise"].values.squeeze()
            return pd.DataFrame({"intercept": intercept, "slopes": slopes, "noise": noise})

    def get_prior_summary(self):
        """
        Get the summary statistics of the prior
        """
        import arviz as az
        import pymc as pm
        # if we've previously used the model for prediction, we need to reset the reference of 'X' to the training data
        if self._predict_done:
            pm.set_data({"X": self._X}, coords={"obs_id": self._X.index, "pred_id": self._X.columns})
        return az.summary(self.trace.prior, var_names = ["intercept", "slopes", "noise"])
  
    def plot_ppc(self, **kwargs):
        """Plot the prior predictive check"""
        import arviz as az
        return az.plot_ppc(self.trace, **kwargs)
    
    def get_posterior(self, return_type = "xarray"):
        """Extracts the prior distribution"""
        import pymc as pm

        assert self._is_fitted,"The model must be fitted before posterior can be returned."
        assert return_type in ["xarray", "numpy", "dataframe"], "return_type must be one of 'xarray', 'numpy' or 'dataframe"

        posterior = self.trace.posterior

        if return_type == "xarray":
            return posterior
        elif return_type == "numpy":
            intercept = posterior["intercept"].stack({"sample":("chain", "draw")}).values
            slopes = posterior["slopes"].stack({"sample":("chain", "draw")}).values
            noise = posterior["noise"].stack({"sample":("chain", "draw")}).values
            return {"intercept": intercept, "slopes": slopes, "noise": noise}
        else:
            intercept = posterior["intercept"].stack({"sample":("chain", "draw")}).values
            slopes = posterior["slopes"].stack({"sample":("chain", "draw")}).values
            noise = posterior["noise"].stack({"sample":("chain", "draw")}).values
            return pd.DataFrame({"intercept": intercept, "slopes": slopes, "noise": noise})
    
    def get_posterior_summary(self):
        """
        Get the summary statistics of the posterior
        """
        import arviz as az
        import pymc as pm
        # if we've previously used the model for prediction, we need to reset the reference of 'X' to the training data
        assert self._is_fitted,"The model must be fitted before posterior summary can be returned."
        return az.summary(self.trace.posterior, var_names = ["intercept", "slopes", "noise"])
    
    def _predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted" = self.is_fitted=True

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            labels predicted for `X`
        """

        assert X.columns.equals(self._X_cols), f"The columns of X must be the same as the columns of the training data: {self._X_cols}"
        y_pred = self._predict_proba(X).mean()
        return y_pred

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
        pred_proba_dist : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """

        import pymc as pm
        from skpro.distributions import Empirical
        
        y_cols = self._y_cols  # columns from y in fit, not automatically stored
        index = X.index
        with self.model:
            # Set the X to be the new 'X' variable and then sample posterior predictive
            pm.set_data({"X": X}, coords={"obs_id": X.index, "pred_id": X.columns})
            self.trace.extend(pm.sample_posterior_predictive(self.trace, random_seed=42, predictions=True))
            self._predict_done = True # a flag
        
        # Extract posterior predictive distributions as an xarray DataArray 
        pred_proba_xarray = self.trace.predictions["y_obs"] 

        # Convert data to pd.DataFrame and format it appropriately for subsequent conversion into a skpro Empirical distribution
        pred_proba_df = pred_proba_xarray.to_dataframe()
        pred_proba_df = pred_proba_df.reset_index()

        # Create a new 'sample_id' column by combining the 'chain' and 'draw' columns
        pred_proba_df["sample_id"] = pred_proba_df["chain"] * self.draws + pred_proba_df["draw"]
        pred_proba_df = pred_proba_df[["obs_id", "sample_id", "y_obs"]]
        pred_proba_df = pred_proba_df.rename(columns = {"y_obs": y_cols[0]})
        pred_proba_df = pred_proba_df.set_index(["sample_id", "obs_id"])

        # Convert data to skpro Empirical distribution
        pred_proba_dist = Empirical(spl=pred_proba_df, index = index, columns = y_cols)
        return pred_proba_dist


    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
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

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        #
        # this can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from skpro or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for most automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params
        # params = {"est": value3, "parama": value4}
        # return params

    def visualize_model(self):
        """
        Use graphviz to visualize the composition of the model
        """
        import pymc as pm
        assert self._is_fitted,"The model must be fitted before visualization can be done."
        return pm.model_to_graphviz(self.model)