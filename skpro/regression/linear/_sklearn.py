"""Adapters to sklearnn linear regressors with probabilistic components."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# based on sktime pipelines

__author__ = ["fkiraly"]

from skpro.regression.adapters.sklearn import SklearnProbaReg
from skpro.regression.base.adapters import _DelegateWithFittedParamForwarding


class ARDRegression(_DelegateWithFittedParamForwarding):
    """ARD regression, direct adapter to sklearn ARDRegression.

    Fit the weights of a regression model, using an ARD prior. The weights of
    the regression model are assumed to be in Gaussian distributions.
    Also estimate the parameters lambda (precisions of the distributions of the
    weights) and alpha (precision of the distribution of the noise).
    The estimation is done by an iterative procedures (Evidence Maximization)

    Parameters
    ----------
    max_iter : int, default=None
        Maximum number of iterations. If `None`, it corresponds to `max_iter=300`.

    tol : float, default=1e-3
        Stop the algorithm if w has converged.

    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.

    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.

    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.

    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.

    compute_score : bool, default=False
        If True, compute the objective function at each step of the model.

    threshold_lambda : float, default=10 000
        Threshold for removing (pruning) weights with high precision from
        the computation.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)

    alpha_ : float
       estimated precision of the noise.

    lambda_ : array-like of shape (n_features,)
       estimated precisions of the weights.

    sigma_ : array-like of shape (n_features, n_features)
        estimated variance-covariance matrix of the weights

    scores_ : float
        if computed, value of the objective function (to be maximized)

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    X_offset_ : float
        If `fit_intercept=True`, offset subtracted for centering data to a
        zero mean. Set to np.zeros(n_features) otherwise.
    """

    def __init__(
        self,
        max_iter=None,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        compute_score=False,
        threshold_lambda=10000.0,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.threshold_lambda = threshold_lambda
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.verbose = verbose

        from sklearn.linear_model import ARDRegression

        skl_estimator = ARDRegression(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            compute_score=compute_score,
            threshold_lambda=threshold_lambda,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose,
        )

        skpro_est = SklearnProbaReg(skl_estimator, inner_type="np.ndarray")
        self._estimator = skpro_est.clone()

        super().__init__()

    FITTED_PARAMS_TO_FORWARD = [
        "coef_",
        "alpha_",
        "lambda_",
        "sigma_",
        "scores_",
        "n_iter_",
        "intercept_",
        "X_offset_",
    ]

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
        param1 = {}
        param2 = {
            "max_iter": 300,
            "tol": 2e-3,
            "alpha_1": 2e-6,
            "alpha_2": 2e-6,
            "lambda_1": 2e-6,
            "lambda_2": 2e-6,
            "compute_score": True,
            "threshold_lambda": 15000.0,
            "fit_intercept": False,
        }
        return [param1, param2]


class BayesianRidge(_DelegateWithFittedParamForwarding):
    """Bayesian ridge regression, direct adapter to sklearn BayesianRidge.

    Fit a Bayesian ridge model. See the Notes section for details on this
    implementation and the optimization of the regularization parameters
    lambda (precision of the weights) and alpha (precision of the noise).

    Parameters
    ----------
    max_iter : int, default=None
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion. If `None`, it
        corresponds to `max_iter=300`.

    tol : float, default=1e-3
        Stop the algorithm if w has converged.

    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.

    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.

    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.

    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.

    alpha_init : float, default=None
        Initial value for alpha (precision of the noise).
        If not set, alpha_init is 1/Var(y).

    lambda_init : float, default=None
        Initial value for lambda (precision of the weights).
        If not set, lambda_init is 1.

    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each iteration of the
        optimization.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        The intercept is not treated as a probabilistic parameter
        and thus has no associated variance. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)

    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        `fit_intercept = False`.

    alpha_ : float
       Estimated precision of the noise.

    lambda_ : float
       Estimated precision of the weights.

    sigma_ : array-like of shape (n_features, n_features)
        Estimated variance-covariance matrix of the weights

    scores_ : array-like of shape (n_iter_+1,)
        If computed_score is True, value of the log marginal likelihood (to be
        maximized) at each iteration of the optimization. The array starts
        with the value of the log marginal likelihood obtained for the initial
        values of alpha and lambda and ends with the value obtained for the
        estimated alpha and lambda.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    X_offset_ : ndarray of shape (n_features,)
        If `fit_intercept=True`, offset subtracted for centering data to a
        zero mean. Set to np.zeros(n_features) otherwise.
    """

    def __init__(
        self,
        max_iter=None,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        alpha_init=None,
        lambda_init=None,
        compute_score=False,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.verbose = verbose

        from sklearn.linear_model import BayesianRidge

        skl_estimator = BayesianRidge(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            alpha_init=alpha_init,
            lambda_init=lambda_init,
            compute_score=compute_score,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose,
        )

        skpro_est = SklearnProbaReg(skl_estimator)
        self._estimator = skpro_est.clone()

        super().__init__()

    FITTED_PARAMS_TO_FORWARD = [
        "coef_",
        "alpha_",
        "lambda_",
        "sigma_",
        "scores_",
        "n_iter_",
        "intercept_",
        "X_offset_",
    ]

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
        param1 = {}
        param2 = {
            "max_iter": 300,
            "tol": 2e-3,
            "alpha_1": 2e-6,
            "alpha_2": 2e-6,
            "lambda_1": 2e-6,
            "lambda_2": 2e-6,
            "compute_score": True,
            "fit_intercept": False,
        }
        return [param1, param2]


class PoissonRegressor(_DelegateWithFittedParamForwarding):
    """Poisson regression, direct adapter to sklearn PoissonRegressor.

    Generalized Linear Model with a Poisson distribution.
    This regressor uses the 'log' link function.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty term. Defaults to 1.0.
        See the notes for the exact mathematical meaning of this
        parameter. alpha = 0 is equivalent to unpenalized GLMs.

    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    
    solver : {'lbfgs', 'newton-cholesky'}, default='lbfgs'
        Algorithm to use in the optimization problem.

    'lbfgs' is an optimization algorithm that approximates the BFGS algorithm
    'newton-cholesky' uses a Newton-CG variant of Newton's method.
        
    max_iter : int, default=100
        The maximal number of iterations for the solver.

    tol : float, default=1e-4
        The convergence tolerance. If it is not None, training will stop
        when (loss > best_loss - tol) for n_iter_no_change consecutive
        epochs.

    verbose : int, default=0
        For the 'sag' and 'lbfgs' solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)

    intercept_ : float
        Independent term in decision function.

    n_iter_ : int
        The actual number of iterations before reaching the stopping criterion.

    n_features_in_ : int
        Number of features seen during :term:'fit'.

    feature_names_in_ : ndarray of shape (n_features,)
        Names of features seen during :term:'fit'.
    """

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True, 
        max_iter=100, 
        tol=1e-4, 
        verbose=0, 
        warm_start=False
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start

        from sklearn.linear_model import PoissonRegressor

        skl_estimator = PoissonRegressor(
            alpha=alpha, 
            fit_intercept=fit_intercept, 
            max_iter=max_iter, 
            tol=tol, 
            verbose=verbose, 
            warm_start=warm_start
        )

        skpro_est = SklearnProbaReg(skl_estimator)
        self._estimator = skpro_est.clone()

        super().__init__()

    FITTED_PARAMS_TO_FORWARD = [
        "coef_", 
        "intercept_", 
        "n_iter_", 
        "n_features_in_", 
        "feature_names_in_"
    ]

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
        param1 = {}
        param2 = {
            "alpha": 2.0,
            "fit_intercept": False,
            "max_iter": 200,
            "tol": 2e-4,
            "verbose": 1,
            "warm_start": True
        }
        return [param1, param2]