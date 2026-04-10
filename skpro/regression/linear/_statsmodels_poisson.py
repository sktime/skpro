"""Adapter to statsmodels Poisson regression with probabilistic predictions."""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd

from skpro.regression.base import BaseProbaRegressor


class StatsmodelsPoissonRegressor(BaseProbaRegressor):
    """Poisson regression, adapter to statsmodels discrete Poisson model.

    Direct interface to ``statsmodels.discrete.discrete_model.Poisson``,
    providing probabilistic predictions via ``skpro.distributions.Poisson``.

    Parameters
    ----------
    add_constant : bool, default=True
        Whether to add an intercept column to the feature matrix.

    offset_var : str or int, default=None
        Column name or index in X to use as offset. Added to linear prediction
        with coefficient 1. Removed from the design matrix before fitting.

    exposure_var : str or int, default=None
        Column name or index in X to use as exposure. ``log(exposure)`` is added
        to linear prediction with coefficient 1. Removed from design matrix.

    missing : str, default='none'
        How to handle missing values. Options: ``'none'``, ``'drop'``, ``'raise'``.

    start_params : array_like, optional, default=None
        Initial guess of the solution for the loglikelihood maximization.

    method : str, default='newton'
        Optimization method passed to ``fit()``.

    maxiter : int, default=35
        Maximum number of iterations for the optimizer.

    tol : float, default=1e-8
        Convergence tolerance for the optimizer.

    disp : bool, default=False
        Whether to print convergence messages.

    cov_type : str, default='nonrobust'
        Covariance type for parameter estimates.

    cov_kwds : dict, optional, default=None
        Extra arguments for covariance calculation.

    Attributes
    ----------
    params_ : ndarray
        Estimated coefficients of the Poisson model.

    pvalues_ : ndarray
        Two-tailed p-values for the estimated parameters.

    bse_ : ndarray
        Standard errors of the estimated parameters.

    llf_ : float
        Log-likelihood of the fitted model.

    aic_ : float
        Akaike information criterion.

    bic_ : float
        Bayesian information criterion.

    nobs_ : float
        Number of observations.

    df_model_ : float
        Model degrees of freedom (number of regressors excluding intercept).

    df_resid_ : float
        Residual degrees of freedom.

    Examples
    --------
    >>> from skpro.regression.linear import StatsmodelsPoissonRegressor
    >>> from sklearn.datasets import make_regression
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> X, _ = make_regression(n_samples=100, n_features=2, noise=0.1)
    >>> X = pd.DataFrame(X, columns=["x1", "x2"])
    >>> rate = np.exp(0.3 * X["x1"] - 0.2 * X["x2"])  # positive Poisson rates
    >>> y = pd.DataFrame(np.random.poisson(rate), columns=["target"])
    >>>
    >>> reg = StatsmodelsPoissonRegressor()
    >>> reg.fit(X, y)  # doctest: +SKIP
    StatsmodelsPoissonRegressor(...)
    >>> y_pred_dist = reg.predict_proba(X)  # doctest: +SKIP
    """

    _tags = {
        "authors": ["Ahmed-Zahran02"],
        "python_dependencies": "statsmodels",
        "capability:multioutput": False,
        "capability:missing": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(
        self,
        add_constant=True,
        offset_var=None,
        exposure_var=None,
        missing="none",
        start_params=None,
        method="newton",
        maxiter=35,
        tol=1e-8,
        disp=False,
        cov_type="nonrobust",
        cov_kwds=None,
    ):
        self.add_constant = add_constant
        self.offset_var = offset_var
        self.exposure_var = exposure_var
        self.missing = missing
        self.start_params = start_params
        self.method = method
        self.maxiter = maxiter
        self.tol = tol
        self.disp = disp
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds

        super().__init__()

    def _prep_x(self, X):
        """Prepare the feature matrix, handling offset/exposure and constant.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        X_out : pd.DataFrame
            Prepared feature matrix (constant added, offset/exposure removed).
        offset_arr : np.ndarray or None
            Offset values extracted from X, if offset_var is set.
        exposure_arr : np.ndarray or None
            Exposure values extracted from X, if exposure_var is set.
        """
        from statsmodels.tools import add_constant

        offset_arr = None
        exposure_arr = None
        cols_to_drop = []

        offset_var = self.offset_var
        exposure_var = self.exposure_var

        if offset_var is not None:
            if isinstance(offset_var, str):
                offset_arr = X[offset_var].to_numpy().flatten()
                cols_to_drop.append(offset_var)
            elif isinstance(offset_var, int):
                col_name = X.columns[offset_var]
                offset_arr = X[col_name].to_numpy().flatten()
                cols_to_drop.append(col_name)

        if exposure_var is not None:
            if isinstance(exposure_var, str):
                exposure_arr = X[exposure_var].to_numpy().flatten()
                cols_to_drop.append(exposure_var)
            elif isinstance(exposure_var, int):
                col_name = X.columns[exposure_var]
                exposure_arr = X[col_name].to_numpy().flatten()
                cols_to_drop.append(col_name)

        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

        if self.add_constant:
            X = add_constant(X, has_constant="add")

        return X, offset_arr, exposure_arr

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            Feature instances to fit regressor to.
        y : pandas DataFrame, must be same length as X
            Labels to fit regressor to.

        Returns
        -------
        self : reference to self
        """
        from statsmodels.discrete.discrete_model import Poisson

        self._y_cols = y.columns

        X_prep, offset_arr, exposure_arr = self._prep_x(X)

        y_inner = y.to_numpy()
        if len(y_inner.shape) > 1 and y_inner.shape[1] == 1:
            y_inner = y_inner[:, 0]

        sm_model = Poisson(
            endog=y_inner,
            exog=X_prep,
            offset=offset_arr,
            exposure=exposure_arr,
            missing=self.missing,
        )

        self._estimator = sm_model

        fit_kwargs = {
            "start_params": self.start_params,
            "method": self.method,
            "maxiter": self.maxiter,
            "tol": self.tol,
            "disp": self.disp,
            "cov_type": self.cov_type,
            "cov_kwds": self.cov_kwds,
        }

        self._fitted_model = sm_model.fit(**fit_kwargs)

        # Forward fitted attributes
        fitted = self._fitted_model
        self.params_ = fitted.params
        self.pvalues_ = fitted.pvalues
        self.bse_ = fitted.bse
        self.llf_ = fitted.llf
        self.aic_ = fitted.aic
        self.bic_ = fitted.bic
        self.nobs_ = fitted.nobs
        self.df_model_ = fitted.df_model
        self.df_resid_ = fitted.df_resid

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
        X_prep, offset_arr, exposure_arr = self._prep_x(X)

        predict_kwargs = {}
        if offset_arr is not None:
            predict_kwargs["offset"] = offset_arr
        if exposure_arr is not None:
            predict_kwargs["exposure"] = exposure_arr

        y_pred = self._fitted_model.predict(X_prep, **predict_kwargs)
        y_pred_df = pd.DataFrame(np.array(y_pred), index=X.index, columns=self._y_cols)
        return y_pred_df

    def _predict_var(self, X):
        """Compute/return variance predictions.

        For a Poisson distribution, variance equals the mean.
        """
        return self._predict(X)

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
        from skpro.distributions.poisson import Poisson

        y_pred = self.predict(X).values
        y_pred_proba = Poisson(mu=y_pred, index=X.index, columns=self._y_cols)
        return y_pred_proba

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        param1 = {}
        param2 = {
            "add_constant": True,
            "method": "bfgs",
            "maxiter": 100,
            "tol": 1e-6,
        }
        param3 = {
            "add_constant": False,
            "method": "bfgs",
        }
        return [param1, param2, param3]
