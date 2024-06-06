"""Simple bayesian linear regression with normal priors; coded on pymc backend"""
# copyright: skpro developers

__author__ = ["meraldoantonio"]

from skpro.regression.base import BaseProbaRegressor

import pandas as pd
import numpy as np

# todo: for imports of skpro soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class BayesianLinearRegressor(BaseProbaRegressor):
    """BayesianLinearRegression with normal priors for slopes and intercept and halfnormal prior for noise.

    Parameters
    ----------
    (to do)
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc

    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
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


    def __init__(self, intercept_sigma=10, slopes_sigma=10, noise_sigma=10, chains=2, draws=2000):

        # priors
        self.intercept_sigma = intercept_sigma
        self.slopes_sigma = slopes_sigma
        self.noise_sigma = noise_sigma
        self.chains = chains
        self.draws = draws

        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)
        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if est.foo == 42:
        #   self.set_tags(handles-missing-data=True)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "handles-missing-data"])

    # todo: implement this, mandatory
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
        self.X = X
        self.y = y
        self.y_vals = y.values[:,0] # we need a 1-dimensional array for compatibility with pymc
        self.X_cols = X.columns
        self._y_cols = y.columns

        with pm.Model() as self.model:

            # Mutable data containers
            X_data = pm.MutableData("X", self.X, dims = ("obs_id", "pred_id"))
            y_data = pm.MutableData("y", self.y_vals, dims = ("obs_id"))

            # Priors for unknown model parameters
            self.intercept = pm.Normal("intercept", mu=0, sigma=self.intercept_sigma)
            self.slopes = pm.Normal("slopes", mu=0, sigma=self.slopes_sigma, shape = X.shape[1], dims=("pred_id"))
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
        # implement logic for prediction here
        # this can read out parameters fitted in fit, or hyperparameters from init
        # no attributes should be written to self

        y_pred = self._predict_proba(X).mean()
        return y_pred

    # todo: implement at least one of the probabilistic prediction methods
    # _predict_proba, _predict_interval, _predict_quantiles
    # if one is implemented, the other two are filled in by default
    # implementation of _predict_proba is preferred, if possible
    #
    # CAVEAT: if not implemented, _predict_proba assumes normal distribution
    # this can be inconsistent with _predict_interval or _predict_quantiles
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
        # if implementing _predict_proba (otherwise delete this method)
        # todo: adapt the following by filling in logic to produce prediction values

        # boilerplate code to create correct output index
        index = X.index
        y_cols = self._y_cols  # columns from y in fit, not automatically stored
        columns = y_cols

        # values = logic to produce prediction values
        # replace this import by the distribution you are using
        # the distribution type can be conditional, e.g., data or parameter dependent
        import pymc as pm
        from skpro.distributions import Empirical

        with self.model:
            # Set the X to be the new 'X' variable and then sample posterior predictive
            pm.set_data({"X": X})
            self.trace.extend(pm.sample_posterior_predictive(self.trace, random_seed=42, predictions=True))
        
        # Note: returns y_obs as xarray.core.dataarray.DataArray containing the posterior predictive samples
        predict_proba = self.trace.predictions["y_obs"] 
        predict_proba_df = predict_proba.to_dataframe()
        predict_proba_df = predict_proba_df.reset_index()
        predict_proba_df["sample_id"] = predict_proba_df["chain"]*self.draws + predict_proba_df["draw"]
        predict_proba_df = predict_proba_df[["obs_id", "sample_id", "y_obs"]]
        predict_proba_df = predict_proba_df.set_index(["sample_id", "obs_id"])
        y_pred = Empirical(spl=predict_proba_df)

        return y_pred


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