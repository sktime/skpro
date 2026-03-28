"""Interface adapter for the Generalized Linear Model Regressor from glum."""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Omswastik-11"]

import numpy as np
import pandas as pd

from skpro.distributions.gamma import Gamma
from skpro.distributions.negative_binomial import NegativeBinomial
from skpro.distributions.normal import Normal
from skpro.distributions.poisson import Poisson
from skpro.regression.base import BaseProbaRegressor


class GlumRegressor(BaseProbaRegressor):
    """Fits a generalized linear model using the glum package.

    Direct interface to glum.GeneralizedLinearRegressor.

    For more information, see:
    https://glum.readthedocs.io/en/latest/index.html

    Parameters
    ----------
    family : str or ExponentialDispersionModel, default='normal'
        The distributional assumption of the GLM.
        One of: 'binomial', 'gamma', 'gaussian', 'inverse.gaussian',
        'normal', 'poisson', 'tweedie', 'negative.binomial'.
    link : str or Link, default='auto'
        The link function of the GLM.
        If 'auto', the canonical link for the family is used.
        Supported links depend on the family. Common options include:
        'identity', 'log', 'logit', 'probit', 'cloglog', 'pow', 'nbinom'.
    alpha : float or array-like, default=None
        Constant that multiplies the penalty terms.
        If ``alpha`` is not None, it must be a non-negative float or an array
        of non-negative floats.
        If ``alpha`` is None, no regularization is applied.
    l1_ratio : float, default=0
        The elastic net mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty (Ridge).
        For ``l1_ratio = 1`` it is an L1 penalty (Lasso).
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor.
    solver : str, default='auto'
        Algorithm to use in the optimization problem.
        'auto' chooses the solver based on the data and model parameters.
        Options include 'lbfgs', 'newton-cg', 'cd', 'trust-constr'.
    max_iter : int, default=100
        The maximal number of iterations for solver algorithms.
    gradient_tol : float, default=None
        Stopping criterion. If the gradient norm is smaller than this threshold,
        the solver stops.
    step_size_tol : float, default=None
        Alternative stopping criterion. If the step size is smaller than this
        threshold, the solver stops.
    hessian_approx : float, default=0.0
        Threshold for updating Hessian. Used in some solvers.
    warm_start : bool, default=False
        Whether to reuse the solution of the previous call to fit as initialization.
    alpha_search : bool, default=False
        Whether to search along the regularization path.
    n_alphas : int, default=100
        Number of alphas along the regularization path.
    min_alpha_ratio : float, default=None
        Length of the path. ``min_alpha_ratio = min_alpha / max_alpha``.
    min_alpha : float, default=None
        Minimum alpha to estimate the model with.
    selection : str, default='cyclic'
        Order of coordinate updates for CD solver.
        'cyclic' or 'random'.
    random_state : int or RandomState, default=None
        Seed for random number generator.
    copy_X : bool, default=None
        Whether to copy X. If False, X may be overwritten.
    check_input : bool, default=True
        Whether to bypass checks on input.
    verbose : int, default=0
        Verbosity level.
    scale_predictors : bool, default=False
        If True, scale all predictors to have standard deviation one.
    lower_bounds : array-like, default=None
        Lower bound for coefficients.
    upper_bounds : array-like, default=None
        Upper bound for coefficients.
    A_ineq : array-like, default=None
        Constraint matrix for linear inequality constraints.
    b_ineq : array-like, default=None
        Constraint vector for linear inequality constraints.
    drop_first : bool, default=False
        If True, drop the first column when encoding categorical variables.
    robust : bool, default=False
        If true, then robust standard errors are computed by default.
    expected_information : bool, default=False
        If true, then the expected information matrix is computed by default.

    Attributes
    ----------
    estimator_ : GeneralizedLinearRegressor
        The fitted glum GeneralizedLinearRegressor estimator.
    dispersion_ : float
        The estimated dispersion parameter.
    """

    _tags = {
        # Packaging information
        # ---------------------
        "authors": [
            "tbenthompson",
            "jtilly",
            "MarcAntoineSchmidtQC",
            "esantorella",
            "lbittarello",
            "stanmart",
            "xhochy",
            "MatthiasSchmidtblaicherQC",
            "Omswastik-11",
        ],
        "maintainers": ["fkiraly", "Omswastik-11"],
        "python_dependencies": "glum",
        "python_version": "<3.14",
        # Estimator type
        # --------------
        "capability:missing": False,
        # CI and test flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        family="normal",
        link="auto",
        alpha=None,
        l1_ratio=0,
        fit_intercept=True,
        solver="auto",
        max_iter=100,
        gradient_tol=None,
        step_size_tol=None,
        hessian_approx=0.0,
        warm_start=False,
        alpha_search=False,
        n_alphas=100,
        min_alpha_ratio=None,
        min_alpha=None,
        selection="cyclic",
        random_state=None,
        copy_X=None,
        check_input=True,
        verbose=0,
        scale_predictors=False,
        lower_bounds=None,
        upper_bounds=None,
        A_ineq=None,
        b_ineq=None,
        drop_first=False,
        robust=False,
        expected_information=False,
    ):
        self.family = family
        self.link = link
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.gradient_tol = gradient_tol
        self.step_size_tol = step_size_tol
        self.hessian_approx = hessian_approx
        self.warm_start = warm_start
        self.alpha_search = alpha_search
        self.n_alphas = n_alphas
        self.min_alpha_ratio = min_alpha_ratio
        self.min_alpha = min_alpha
        self.selection = selection
        self.random_state = random_state
        self.copy_X = copy_X
        self.check_input = check_input
        self.verbose = verbose
        self.scale_predictors = scale_predictors
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.drop_first = drop_first
        self.robust = robust
        self.expected_information = expected_information

        super().__init__()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"family": "normal"}
        params2 = {"family": "gamma", "link": "log"}
        params3 = {"family": "poisson"}
        params4 = {"family": "negative.binomial"}
        params5 = {"family": "normal", "alpha": 0.1, "l1_ratio": 0.5}
        return [params1, params2, params3, params4, params5]

    def _fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self
        """
        from glum import GeneralizedLinearRegressor

        self.estimator_ = GeneralizedLinearRegressor(
            family=self.family,
            link=self.link,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
            max_iter=self.max_iter,
            gradient_tol=self.gradient_tol,
            step_size_tol=self.step_size_tol,
            hessian_approx=self.hessian_approx,
            warm_start=self.warm_start,
            alpha_search=self.alpha_search,
            n_alphas=self.n_alphas,
            min_alpha_ratio=self.min_alpha_ratio,
            min_alpha=self.min_alpha,
            selection=self.selection,
            random_state=self.random_state,
            copy_X=self.copy_X,
            check_input=self.check_input,
            verbose=self.verbose,
            scale_predictors=self.scale_predictors,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            A_ineq=self.A_ineq,
            b_ineq=self.b_ineq,
            drop_first=self.drop_first,
            robust=self.robust,
            expected_information=self.expected_information,
        )

        self.estimator_.fit(X, np.ravel(y))

        self._y_cols = y.columns

        # Estimate dispersion
        mu = self.estimator_.predict(X)
        self.dispersion_ = self.estimator_.family_instance.dispersion(np.ravel(y), mu)

        return self

    def _predict(self, X):
        """Predict mean.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted mean.
        """
        y_pred = self.estimator_.predict(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(y_pred, index=X.index, columns=self._y_cols)
        return y_pred

    def _predict_var(self, X):
        """Predict variance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_var : array-like of shape (n_samples,)
            The predicted variance.
        """
        mu = self.estimator_.predict(X)
        var = self.estimator_.family_instance.variance(mu, dispersion=self.dispersion_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(var, index=X.index, columns=self._y_cols)
        return var

    def _predict_proba(self, X):
        """Predict distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        dist : skpro.distributions.BaseDistribution
            The predicted distribution.
        """
        mu = self._predict(X)
        family = self.family

        if isinstance(family, str):
            family_str = family.lower()
        else:
            # If family is an object, we need to infer the type
            # This is tricky, but let's assume string for now as per init
            family_str = str(family).lower()

        if "normal" in family_str or "gaussian" in family_str:
            # Normal distribution
            # Variance = dispersion * v(mu) = dispersion * 1 = dispersion
            # So sigma = sqrt(dispersion)
            sigma = np.sqrt(self.dispersion_)
            return Normal(mu=mu, sigma=sigma, index=X.index, columns=self._y_cols)

        elif "poisson" in family_str:
            # Poisson distribution
            # skpro Poisson takes mu.
            # If dispersion != 1, it's not standard Poisson.
            # But skpro Poisson is standard.
            return Poisson(mu=mu, index=X.index, columns=self._y_cols)

        elif "gamma" in family_str:
            # Gamma distribution
            # mu = alpha / beta
            # var = alpha / beta^2 = dispersion * mu^2
            # alpha = 1 / dispersion
            # beta = 1 / (dispersion * mu)
            alpha = 1.0 / self.dispersion_
            beta = 1.0 / (self.dispersion_ * mu)
            return Gamma(alpha=alpha, beta=beta, index=X.index, columns=self._y_cols)

        elif "negative.binomial" in family_str:
            # Negative Binomial
            # var = mu + theta * mu^2
            # skpro NB takes mu and alpha (where var = mu + mu^2/alpha)
            # So alpha_skpro = 1/theta_glum

            # We need to extract theta from family string or object
            # If family is string like 'negative.binomial(1.5)', theta is 1.5
            # If family is 'negative.binomial', theta is default 1.0?

            theta = 1.0
            if "(" in family_str:
                try:
                    theta = float(family_str.split("(")[1].split(")")[0])
                except ValueError:
                    pass

            # Also check if family_instance has theta
            if hasattr(self.estimator_.family_instance, "theta"):
                theta = self.estimator_.family_instance.theta

            alpha = 1.0 / theta
            return NegativeBinomial(
                mu=mu,
                alpha=alpha,
                index=X.index,
                columns=self._y_cols,
            )

        else:
            raise NotImplementedError(
                f"Distribution for family '{family}' not implemented in "
                "skpro interface."
            )
