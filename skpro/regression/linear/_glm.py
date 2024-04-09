"""Interface adapter for the Generalized Linear Model Regressor with Gaussian Link."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import pandas as pd

from skpro.regression.base import BaseProbaRegressor


class GLMRegressor(BaseProbaRegressor):
    """
    Fits a generalized linear model with a gaussian link.

    Direct interface to ``statsmodels.genmod.generalized_linear_model.GLM``
    from the ``statsmodels`` package.

    For a direct link to statmodels' Generalized Linear Models module see:
    https://www.statsmodels.org/stable/glm.html#module-reference

    Parameters
    ----------
    missing : str
        Available options are 'none', 'drop' and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default = 'none'

    start_params : array_like (optional)
        Initial guess of the solution for the loglikelihood maximization.
        The default is family-specific and is given by the
        family.starting_mu(endog). If start_params is given then the initial
        mean will be calculated as np.dot(exog, start_params).
        This parameter is used inside the GLM fit() function.

    maxiter : int, optional, default=100
        Number of iterations. This parameter is used inside the GLM fit() function.

    method : str, optional, default='IRLS'
        Default is 'IRLS' for iteratively re-weighted least squares.
        This parameter is used inside the GLM fit() function.

    tol : float, optional, default=1e-8
        Convergence tolerance. Default is 1e-8. This parameter is
        used inside the GLM fit() function.

    scale : str/float, optional, default=None
        scale can be 'X2', 'dev', or a float. The default value is None,
        which uses X2 for gamma, gaussian and inverse gaussian. X2 is
        Pearson's chi-square divided by df_resid. The default is 1 for
        the Bionmial and Poisson families. dev is the deviance divided
        by df_resid. This parameter is used inside the GLM fit() function.

    cov_type : str, optional, default='nonrobust'
        The type of parameter estimate covariance matrix to compute.
        This parameter is used inside the GLM fit() function.

    cov_kwds : dict-like, optional, default=None
        Extra arguments for calculating the covariance of the
        parameter estimates. This parameter is used inside the GLM fit() function.

    use_t : bool, optional, default=False
        if True, the Student t-distribution if used for inference.
        This parameter is used inside the GLM fit() function.

    full_output : bool, optional, default=True
        Set to True to have all available output in the Results object’s
        mle_retvals attribute. The output is dependent on the solver. See
        LikelihoodModelResults notes section for more information. Not used
        if methhod is IRLS. This parameter is used inside the GLM fit() function.

    disp : bool, optional, default=False
        Set to True to print convergence messages. Not used if method
        is IRLS. This parameter is used inside the GLM fit() function.

    max_start_irls : int, optional, default=3
        The number of IRLS iterations used to obtain starting values for
        gradient optimization. Only relevenat if method is set to something
        other than "IRLS". This parameter is used inside the GLM fit() function.

    add_constant : bool, optional, default=False
        statsmodels does not include an intercept by default. Specify this as
        True if you would like to add an intercept (floats of 1s) to the
        dataset X. Default = False. Note that when the input is a pandas
        Series or DataFrame, the added column's name is 'const'.

    Attributes
    ----------
    df_model_ : float
        Model degrees of freedom is equal to p - 1, where p is the number of
        regressors. Note that the intercept is not reported as a degree of freedom.

    df_resid_ : float
        Residual degrees of freedom is equal to the number of observation n
        minus the number of regressors p.

    iteration_ : int
        The number of iterations that fit has run. Initialized at 0.
        Only available after fit is called

    mu_ : ndarray
        The mean response of the transformed variable. mu is the value of the
        inverse of the link function at lin_pred, where lin_pred is the linear
        predicted value of the WLS fit of the transformed variable. mu is only
        available after fit is called. See statsmodels.families.family.fitted
        of the distribution family for more information.

    n_trials_ : ndarray
        Note that n_trials is a reference to the data so that if data is
        already an array and it is changed, then n_trials changes as well.
        n_trials is the number of binomial trials and only available with that
        distribution. See statsmodels.families.Binomial for more information.

    scale_ : float
        The estimate of the scale / dispersion of the model fit.
        Only available after fit is called. See GLM.fit and GLM.estimate_scale
        for more information.

    scaletype_ : str
        The scaling used for fitting the model. This is only available
        after fit is called. The default is None. See GLM.fit for
        more information.

    weights_ : ndarray
        The value of the weights after the last iteration of fit.
        Only available after fit is called. See statsmodels.families.family
        for the specific distribution weighting functions.

    glm_fit_ : GLM
        fitted generalized linear model

    fit_history_ : dict
        Contains information about the iterations.
        Its keys are iterations, deviance and params. Only available after
        fit is called.

    model_ : class instance
        Pointer to GLM model instance that called fit.

    nobs_ : float
        The number of observations n. Only available after fit is called.

    normalized_cov_params_ : ndarray
        For Gaussian link: This is the p x p normalized covariance of the
        design / exogenous data. This is approximately equal to (X.T X)^(-1)

    params_ : ndarray
        The coefficients of the fitted model. Note that interpretation of the
        coefficients often depends on the distribution family and the data.

    pvalues_ : ndarray
        The two-tailed p-values for the parameters.

    scale_ : float
        The estimate of the scale / dispersion for the model fit.
        See GLM.fit and GLM.estimate_scale for more information.

    stand_errors_ : ndarray
        The standard errors of the fitted GLM.
    """

    _tags = {
        "authors": ["julian-fong"],
        "maintainers": ["julian-fong"],
        "python_version": None,
        "python_dependencies": "statsmodels",
        "capability:multioutput": False,
        "capability:missing": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(
        self,
        missing="none",
        start_params=None,
        maxiter=100,
        method="IRLS",
        tol=1e-8,
        scale=None,
        cov_type="nonrobust",
        cov_kwds=None,
        use_t=None,
        full_output=True,
        disp=False,
        max_start_irls=3,
        add_constant=False,
    ):
        super().__init__()
        from statsmodels.genmod.families.family import Gaussian

        self._family = Gaussian()
        self.missing = missing
        self.start_params = start_params
        self.maxiter = maxiter
        self.method = method
        self.tol = tol
        self.scale = scale
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.use_t = use_t
        self.full_output = full_output
        self.disp = disp
        self.max_start_irls = max_start_irls
        self.add_constant = add_constant

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            A (n,k) array where n is the number of observations and k is the number
            of regressors. An intercept is not included by default and should be
            added by the user (models specified using a formula include an
            intercept by default). Equivalent to statsmodel's (exog).

        y : pandas DataFrame
            1d array of the endogenous response variable. This array can be 1d or
            2d. Binomial family models accept a 2d array  with two columns.
            If supplied each observation is expected to  be [success, failure].
            Equivalent to statsmodel's (endog).

        Returns
        -------
        self : reference to self
        """
        from statsmodels.genmod.generalized_linear_model import GLM

        X_ = self._prep_x(X)

        y_col = y.columns

        glm_estimator = GLM(
            endog=y,
            exog=X_,
            family=self._family,
            missing=self.missing,
        )

        self._estimator = glm_estimator

        fitted_glm_model = glm_estimator.fit(
            self.start_params,
            self.maxiter,
            self.method,
            self.tol,
            self.scale,
            self.cov_type,
            self.cov_kwds,
            self.use_t,
            self.full_output,
            self.disp,
            self.max_start_irls,
        )

        PARAMS_TO_FORWARD = {
            "df_model_": glm_estimator.df_model,
            "df_resid_": glm_estimator.df_resid,
            "mu_": glm_estimator.mu,
            "n_trials_": glm_estimator.n_trials,
            "weights_": glm_estimator.weights,
            "scaletype_": glm_estimator.scaletype,
        }

        for k, v in PARAMS_TO_FORWARD.items():
            setattr(self, k, v)

        # forward some parameters to self
        FITTED_PARAMS_TO_FORWARD = {
            "glm_fit_": fitted_glm_model,
            "y_col": y_col,
            "fit_history_": fitted_glm_model.fit_history,
            "iteration_": fitted_glm_model.fit_history["iteration"],
            "model_": fitted_glm_model.model,
            "nobs_": fitted_glm_model.nobs,
            "normalized_cov_params_": fitted_glm_model.normalized_cov_params,
            "params_": fitted_glm_model.params,
            "pvalues_": fitted_glm_model.pvalues,
            "scale_": fitted_glm_model.scale,
            "stand_errors_": fitted_glm_model.bse,
        }

        for k, v in FITTED_PARAMS_TO_FORWARD.items():
            setattr(self, k, v)

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted"

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`, with same columns as y in fit
        """
        X_ = self._prep_x(X)

        index = X_.index
        y_column = self.y_col
        y_pred_series = self.glm_fit_.predict(X_)
        y_pred = pd.DataFrame(y_pred_series, index=index, columns=y_column)

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
        y_pred : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        from skpro.distributions.normal import Normal

        X_ = self._prep_x(X)

        # instead of using the conventional predict() method, we use statsmodels
        # get_prediction method, which returns a pandas df that contains
        # the prediction and prediction variance i.e mu and sigma
        y_column = self.y_col
        y_predictions_df = self.glm_fit_.get_prediction(X_).summary_frame()
        y_mu = y_predictions_df["mean"].rename("mu").to_frame()
        y_sigma = y_predictions_df["mean_se"].rename("sigma").to_frame()
        params = {
            "mu": y_mu,
            "sigma": y_sigma,
            "index": X_.index,
            "columns": y_column,
        }
        y_pred = Normal(**params)
        return y_pred

    def _prep_x(self, X):
        """
        Return a copy of X with an added constant of self.add_constant = True.

        Parameters
        ----------
        X : pandas DataFrame
            Dataset that the user is trying to do inference on

        Returns
        -------
        X_ : pandas DataFrame
            A copy of the input X with an added column 'const' with is an
            array of len(X) of 1s
        """
        from statsmodels.tools import add_constant

        if self.add_constant:
            X_ = add_constant(X)
            return X_
        else:
            return X

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
        params1 = {}
        params2 = {"add_constant": True}

        return [params1, params2]
