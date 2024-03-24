"""Interface adapter for the Generalized Linear Model Regressor with Gaussian Link"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd

from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Gaussian

class GaussianRegressor(BaseProbaRegressor):
    """
    Fits a generalized linear model with a gaussian link.

    Direct interface to ``statsmodels.genmod.generalized_linear_model.GLM`` 
    from the ``statsmodels`` package.

    statsmodels uses parameters 'exog' and 'endog' to denote the X and y values
    respectively and supports two separate definition of weights: frequency 
    and variance.
    
    For a direct link to statmodels' Generalized Linear Models module see:
    https://www.statsmodels.org/stable/glm.html#module-reference

    Parameters
    ----------
    endog : pandas DataFrame
        1d array of the endogenous (y) response variable. This array can be 1d
        or 2d. Binomial family models accept a 2d array with two columns. If supplied
        each observation is expected to be [success, failure].

    exog : pandas DataFrame
        A (n,k) array where n is the number of observations and k is the number
        of regressors. An intercept is not included by default and should be 
        added by the user (models specified using a formula include an 
        intercept by default).

    family : family class instance
        To specify the binomial distribution family = sm.family.Binomial() Each
        family can take a link instance as an argument. 
        See statsmodels.family.family for more information.

    offset : array_like or None
        An offset to be included in the model. If provided, must be an array
        whose length is the number of rows in exog (x).

    exposure : array_like or None
        Log(exposure) will be added to the linear prediction in the model. 
        Exposure is only valid if the log link is used. If provided, it must be
        an array with the same length as endog (y).
    
    freq_weights : array_like
        1d array of frequency weights. The default is None. If None is selected
        or a blank value, then the algorithm will replace with an array of 1s 
        with length equal to the endog.
    
    var_weights : array_like
        1d array of variance (analytic) weights. The default is None. If None 
        is selected or a blank value, then the algorithm will replace with an 
        array of 1s with length equal to the endog.
    missing : str
        Available options are 'none', 'drop' and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default = 'none'

    Attributes
    ----------
    df_model : float
        Model degrees of freedom is equal to p - 1, where p is the number of 
        regressors. Note that the intercept is not reported as a degree of freedom.

    df_resid : float
        Residual degrees of freedom is equal to the number of observation n 
        minus the number of regressors p.

    endog : pandas DataFrame
        Note that endog is a reference to the data so that if data is already 
        an array and it is changed, then endog changes as well.

    exposure : array_like
        Include ln(exposure) in model with coefficient constrained to 1. 
        Can only be used if the link is the logarithm function.

    exog : pandas DataFrame
        Note that exog is a reference to the data so that if data is already 
        an array and it is changed, then exog changes as well.

    freq_weights : ndarray
        Note that freq_weights is a reference to the data so that if data 
        is already an array and it is changed, then freq_weights changes 
        as well.

    var_weights : ndarray
        Note that var_weights is a reference to the data so that if 
        data is already an array and it is changed, then var_weights 
        changes as well.

    iteration : int
        The number of iterations that fit has run. Initialized at 0.

    family : family class instance
        he distribution family of the model. Can be any family 
        in statsmodels.families. Default is Gaussian.

    mu : ndarray
        The mean response of the transformed variable. mu is the value of the 
        inverse of the link function at lin_pred, where lin_pred is the linear 
        predicted value of the WLS fit of the transformed variable. mu is only 
        available after fit is called. See statsmodels.families.family.fitted 
        of the distribution family for more information.

    n_trials : ndarray
        Note that n_trials is a reference to the data so that if data is 
        already an array and it is changed, then n_trials changes as well. 
        n_trials is the number of binomial trials and only available with that 
        distribution. See statsmodels.families.Binomial for more information.

    normalized_cov_params : ndarray
        The p x p normalized covariance of the design / exogenous data. This 
        is approximately equal to (X.T X)^(-1)

    offset : array_like
        Include offset in model with coefficient constrained to 1.

    scale : float
        The estimate of the scale / dispersion of the model fit. 
        Only available after fit is called. See GLM.fit and GLM.estimate_scale 
        for more information.

    scaletype : str
        The scaling used for fitting the model. This is only available 
        after fit is called. The default is None. See GLM.fit for 
        more information.

    weights : ndarray
        The value of the weights after the last iteration of fit. 
        Only available after fit is called. See statsmodels.families.family 
        for the specific distribution weighting functions.
    """

    _tags = {
        "authors": ["julian-fong"],
        "maintainers": ["julian-fong"],
        "python_version": None,
        "python_dependencies": None,
        "capability:multioutput": False,
        "capability:missing": False,
        "X_inner_mtype": "pd_DataFrame",
        "y_inner_mtype": "pd_DataFrame",
    }

    def __init__(
            self,
            endog,
            exog,
            family = None,
            offset = None,
            exposure = None,
            freq_weights = None,
            var_weights = None,
            missing = "none",
    ):
        self.endog = endog,
        self.exog = exog,
        self.family = Gaussian(),
        self.offset = offset,
        self.exposure = exposure,
        self.freq_weights = freq_weights,
        self.var_weights = var_weights,
        self.missing = missing

        super().__init__()

        glm_estimator = GLM(
            endog = endog,
            exog = exog,
            family = family,
            offset = offset,
            exposure = exposure,
            freq_weights = freq_weights,
            var_weights = var_weights,
            missing = missing
        )

        self._estimator = glm_estimator #does this need to be cloned using some clone method?

    def _fit(
            self, 
            start_params=None, 
            maxiter=100, 
            method='IRLS', 
            tol=1e-8,
            scale=None, 
            cov_type='nonrobust', 
            cov_kwds=None, 
            use_t=None,
            full_output=True, 
            disp=False,
            max_start_irls=3
        ):
        """
        Fits the regressor to the data. 
        
        Note that the parameters X, y were defined when calling the statsmodel 
        GLM constructer.
                
        Writes to self:
            Sets fitted model attributes ending in "_".
        
        Parameters
        ----------
        start_params : array_like (optional)
            Initial guess of the solution for the loglikelihood maximization. 
            The default is family-specific and is given by the 
            family.starting_mu(endog). If start_params is given then the initial 
            mean will be calculated as np.dot(exog, start_params).

        maxiter : int
            Number of iterations

        method : str
            Default is 'IRLS' for iteratively re-weighted least squares

        tol : float
            Convergence tolerance. Default is 1e-8

        scale : str/float 
            scale can be 'X2', 'dev', or a float. The default value is None,
            which uses X2 for gamma, gaussian and inverse gaussian. X2 is
            Pearson's chi-square divided by df_resid. The default is 1 for
            the Bionmial and Poisson families. dev is the deviance divided
            by df_resid

        cov_type : str
            The type of parameter estimate covariance matrix to compute

        cov_kwds : dict-like
            Extra arguments for calculating the covariance of the 
            parameter estimates
        
        use_t : bool 
            if True, the Student t-distribution if used for inference

        full_output : bool
            Set to True to have all available output in the Results object’s 
            mle_retvals attribute. The output is dependent on the solver. See 
            LikelihoodModelResults notes section for more information. Not used
            if methhod is IRLS.
        
        disp : bool
            Set to True to print convergence messages. Not used if method 
            is IRLS
        
        max_start_irls : int
            The number of IRLS iterations used to obtain starting values for
            gradient optimization. Only relevenat if method is set to something
            other than "IRLS"

        Returns
        -------
        self : reference to self
        """

        fitted_glm_model = self._estimator.fit(
            start_params,
            maxiter,
            method,
            tol,
            scale,
            cov_type,
            cov_kwds,
            use_t,
            full_output,
            disp,
            max_start_irls,
        )

        FITTED_PARAMS_TO_FORWARD = ["glm_estimator_"]

        for param in FITTED_PARAMS_TO_FORWARD:
            setattr(self, param, fitted_glm_model)

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
        y_column = self.endog.columns
        y_pred_series = self.glm_estimator_.predict(X)
        y_pred = pd.DataFrame(y_pred_series, columns = [y_column])

        return y_pred
    

    def _predict_proba(self, X):
        pass

