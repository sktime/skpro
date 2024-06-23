"""Interface adapter for the Generalized Linear Model Regressor."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ShreeshaM07", "julian-fong"]

import numpy as np
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
    family : string, default : "Normal"
        The family parameter denotes the type of distribution
        that will be used.
        Available family/distributions are
        1."Normal"
        2."Poisson"
        3."Gamma"
    link : string, default : None
        This parameter is used to represent the link function to be
        used with the distribution.
        If default is None it will internally replace with default of the
        respective family. The default is the first string
        against each family below.
        Available safe options for the respective family are:
        ``Normal`` : "Identity", "Log", "InversePower";
        ``Poisson`` : "Log", "Identity", "Sqrt";
        ``Gamma`` : "InversePower", "Log", "Identity";
    offset_var : string or int, default = None
        Pass the column name as a string or column number as an int in X.
        If string, then the exog or ``X`` passed while ``fit``-ting
        must have an additional column with column name passed through
        ``offset_var`` with any values against each row. When ``predict``ing
        have an additional column with name same as string passed through ``offset_var``
        in X with all the ``offset_var`` values for predicting
        stored in the column for each row.
        If ``int`` it corresponding column number will be considered.
    exposure_var : string or int, default = None
        Pass the column name as a string or column number as an int in X.
        If string, then the exog or ``X`` passed while ``fit``-ting
        must have an additional column with column name passed through
        ``exposure_var`` with any values against each row. When ``predict``ing
        have additional column with name same as string passed through ``exposure_var``
        in X with all the ``exposure_var`` values for predicting
        stored in the column for each row.
        If ``int`` it corresponding column number will be considered.
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
        "authors": ["ShreeshaM07", "julian-fong"],
        "maintainers": ["ShreeshaM07", "julian-fong"],
        "python_version": None,
        "python_dependencies": "statsmodels",
        "capability:multioutput": False,
        "capability:missing": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def _str_to_sm_family(self, family, link):
        """Convert the string to a statsmodel object.

        If the link function is also explcitly mentioned then include then
        that must be passed to the family/distribution object.
        """
        from warnings import warn

        from statsmodels.genmod.families.family import Gamma, Gaussian, Poisson
        from statsmodels.genmod.families.links import Identity, InversePower, Log, Sqrt

        sm_fmly = {
            "Normal": Gaussian,
            "Poisson": Poisson,
            "Gamma": Gamma,
        }

        links = {
            "Log": Log,
            "Identity": Identity,
            "InversePower": InversePower,
            "Sqrt": Sqrt,
        }

        if link in links:
            link_function = links[link]()
            try:
                return sm_fmly[family](link_function)
            except Exception:
                msg = "Invalid link for family, default link will be used"
                warn(msg)

        return sm_fmly[family]()

    # TODO (release 2.4.0)
    # replace the existing definition of `__init__` with
    # the below definition for `__init__`.
    # def __init__(
    #     self,
    #     family="Normal",
    #     link=None,
    #     offset_var=None,
    #     exposure_var=None,
    #     missing="none",
    #     start_params=None,
    #     maxiter=100,
    #     method="IRLS",
    #     tol=1e-8,
    #     scale=None,
    #     cov_type="nonrobust",
    #     cov_kwds=None,
    #     use_t=None,
    #     full_output=True,
    #     disp=False,
    #     max_start_irls=3,
    #     add_constant=False,
    # ):
    #     super().__init__()

    #     self.family = family
    #     self.link = link
    #     self.offset_var = offset_var
    #     self.exposure_var = exposure_var
    #     self.missing = missing
    #     self.start_params = start_params
    #     self.maxiter = maxiter
    #     self.method = method
    #     self.tol = tol
    #     self.scale = scale
    #     self.cov_type = cov_type
    #     self.cov_kwds = cov_kwds
    #     self.use_t = use_t
    #     self.full_output = full_output
    #     self.disp = disp
    #     self.max_start_irls = max_start_irls
    #     self.add_constant = add_constant

    #     self._family = self.family
    #     self._link = self.link
    #     self._offset_var = self.offset_var
    #     self._exposure_var = self.exposure_var
    #     self._missing = self.missing
    #     self._start_params = self.start_params
    #     self._maxiter = self.maxiter
    #     self._method = self.method
    #     self._tol = self.tol
    #     self._scale = self.scale
    #     self._cov_type = self.cov_type
    #     self._cov_kwds = self.cov_kwds
    #     self._use_t = self.use_t
    #     self._full_output = self.full_output
    #     self._disp = self.disp
    #     self._max_start_irls = self.max_start_irls
    #     self._add_constant = self.add_constant

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
        family="Normal",
        link=None,
        offset_var=None,
        exposure_var=None,
    ):
        # The default values of the parameters
        # are replaced with the changed sequence
        # of parameters ranking for each of them
        # from 0 to 16(total 17 parameters).
        super().__init__()

        self.family = family
        self.link = link
        self.offset_var = offset_var
        self.exposure_var = exposure_var
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

        self._family = self.family
        self._link = self.link
        self._offset_var = self.offset_var
        self._exposure_var = self.exposure_var
        self._missing = self.missing
        self._start_params = self.start_params
        self._maxiter = self.maxiter
        self._method = self.method
        self._tol = self.tol
        self._scale = self.scale
        self._cov_type = self.cov_type
        self._cov_kwds = self.cov_kwds
        self._use_t = self.use_t
        self._full_output = self.full_output
        self._disp = self.disp
        self._max_start_irls = self.max_start_irls
        self._add_constant = self.add_constant

        from warnings import warn

        warn(
            "Note: in `GLMRegressor`, the sequence of the parameters will change "
            "in skpro version 2.5.0. It will be as per the order present in the"
            "current docstring with the top one being the first parameter.\n"
            "The defaults for the parameters will remain same and "
            "there will be no changes.\n"
            "Please use the `kwargs` calls instead of positional calls for the"
            "parameters until the release of skpro 2.5.0 "
            "as this will avoid any discrepancies."
        )

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

        # remove the offset and exposure columns which
        # was inserted to maintain the shape
        offset_var = self._offset_var
        exposure_var = self._exposure_var

        X_ = self._prep_x(X, offset_var, exposure_var, False)

        y_col = y.columns

        family = self._family
        link = self._link
        sm_family = self._str_to_sm_family(family=family, link=link)

        glm_estimator = GLM(
            endog=y,
            exog=X_,
            family=sm_family,
            missing=self._missing,
        )

        self._estimator = glm_estimator

        glm_fit_params = {
            "start_params": self._start_params,
            "maxiter": self._maxiter,
            "method": self._method,
            "tol": self._tol,
            "scale": self._scale,
            "cov_type": self._cov_type,
            "cov_kwds": self._cov_kwds,
            "use_t": self._use_t,
            "full_output": self._full_output,
            "disp": self._disp,
            "max_start_irls": self._max_start_irls,
        }

        fitted_glm_model = glm_estimator.fit(**glm_fit_params)

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
        offset_var = self._offset_var
        exposure_var = self._exposure_var
        offset_arr = None
        exposure_arr = None

        X_, offset_arr, exposure_arr = self._prep_x(X, offset_var, exposure_var, True)

        index = X_.index
        y_column = self.y_col
        y_pred_series = self.glm_fit_.predict(
            X_, offset=offset_arr, exposure=exposure_arr
        )
        y_pred = pd.DataFrame(y_pred_series, index=index, columns=y_column)

        return y_pred

    def _params_sm_to_skpro(self, y_predictions_df, index, columns, family):
        """Convert the statsmodels output to equivalent skpro distribution."""
        from skpro.distributions.gamma import Gamma
        from skpro.distributions.normal import Normal
        from skpro.distributions.poisson import Poisson

        skpro_distr = {
            "Normal": Normal,
            "Poisson": Poisson,
            "Gamma": Gamma,
        }

        params = {}
        skp_dist = Normal

        if family in skpro_distr:
            skp_dist = skpro_distr[family]

        if skp_dist == Normal:
            y_mu = y_predictions_df["mean"].rename("mu").to_frame()
            y_sigma = y_predictions_df["mean_se"].rename("sigma").to_frame()
            params["mu"] = y_mu
            params["sigma"] = y_sigma
        elif skp_dist == Poisson:
            y_mu = y_predictions_df["mean"].rename("mu").to_frame()
            params["mu"] = y_mu
        elif skp_dist == Gamma:
            y_mean = y_predictions_df["mean"]
            y_sd = y_predictions_df["mean_se"]
            y_alpha = (y_mean / y_sd) ** 2
            y_beta = (y_mean / (y_sd**2)).rename("beta").to_frame()
            y_alpha = y_alpha.rename("alpha").to_frame()
            params["alpha"] = y_alpha
            params["beta"] = y_beta

        params["index"] = index
        params["columns"] = columns

        y_pred = skp_dist(**params)
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
        # remove the offset and exposure columns
        # which was inserted to maintain the shape
        offset_var = self._offset_var
        exposure_var = self._exposure_var

        X_ = self._prep_x(X, offset_var, exposure_var, False)

        # instead of using the conventional predict() method, we use statsmodels
        # get_prediction method, which returns a pandas df that contains
        # the prediction and prediction variance i.e mu and sigma
        y_column = self.y_col
        y_predictions_df = self.glm_fit_.get_prediction(X_).summary_frame()

        # convert the returned values to skpro equivalent distribution
        family = self._family
        index = X_.index
        columns = y_column

        y_pred = self._params_sm_to_skpro(y_predictions_df, index, columns, family)
        return y_pred

    def _prep_x(self, X, offset_var, exposure_var, rtn_off_exp_arr):
        """
        Return a copy of X with an added constant of self.add_constant = True.

        If rtn_off_exp_arr is True it will also return offset and exposure
        arrays along with updated X.

        Parameters
        ----------
        X : pandas DataFrame
            Dataset that the user is trying to do inference on

        Returns
        -------
        X_ : pandas DataFrame
            A copy of the input X with an added column 'const' with is an
            array of len(X) of 1s
        offset_arr : numpy.array
            The copy of column which is meant for offsetting present in X.
        exposure_arr : numpy.array
            The copy of column which is meant for exposure present in X.
        """
        from statsmodels.tools import add_constant

        offset_arr = None
        exposure_arr = None
        if offset_var is not None:
            if isinstance(offset_var, str):
                offset_var = pd.Index([offset_var])
                offset_arr = np.array(X[offset_var]).flatten()
            elif isinstance(offset_var, int):
                offset_arr = np.array(X.iloc[:, offset_var]).flatten()
                offset_var = pd.Index([X.iloc[:, offset_var].name])
        if exposure_var is not None:
            if isinstance(exposure_var, str):
                exposure_var = pd.Index([exposure_var])
                exposure_arr = np.array(X[exposure_var]).flatten()
            elif isinstance(exposure_var, int):
                exposure_arr = np.array(X.iloc[:, exposure_var]).flatten()
                exposure_var = pd.Index([X.iloc[:, exposure_var].name])
        # drop the offset and exposure columns from X
        columns_to_drop = []
        if offset_var is not None:
            columns_to_drop.append(offset_var[0])
        if exposure_var is not None:
            columns_to_drop.append(exposure_var[0])
        if columns_to_drop:
            X = X.drop(columns_to_drop, axis=1)

        if self._add_constant:
            X_ = add_constant(X)
            if rtn_off_exp_arr:
                return X_, offset_arr, exposure_arr
            return X_
        else:
            if rtn_off_exp_arr:
                return X, offset_arr, exposure_arr
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
        params3 = {
            "family": "Poisson",
            "add_constant": True,
        }
        params4 = {"family": "Gamma"}
        params5 = {
            "family": "Normal",
            "link": "InversePower",
        }
        params6 = {
            "family": "Poisson",
            "link": "Log",
            "add_constant": True,
        }

        return [params1, params2, params3, params4, params5, params6]
