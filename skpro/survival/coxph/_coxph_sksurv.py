"""Interface adapters to scikit-survival Cox PH model."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from skpro.survival.adapters.sksurv import _SksurvAdapter
from skpro.survival.base import BaseSurvReg


class CoxPHSkSurv(_SksurvAdapter, BaseSurvReg):
    """Cox proportional hazards, from scikit-survival.

    Direct interface to ``sksurv.linear_model.CoxPHSurvivalAnalysis``, by ``sebp``.

    Parameters
    ----------
    alpha : float, or ndarray of shape (n_features,), optional, default: 0
        Regularization parameter for ridge regression penalty.
        If a single float, the same penalty is used for all features.
        If an array, there must be one penalty for each feature.
        If you want to include a subset of features without penalization,
        set the corresponding entries to 0.

    ties : {'breslow', 'efron'}, optional, default: 'breslow'
        The method to handle tied event times. If there are
        no tied event times all the methods are equivalent.

    n_iter : int, optional, default: 100
        Maximum number of iterations.

    tol : float, optional, default: 1e-9
        Convergence criteria. Convergence is based on the negative log-likelihood:

        |1 - (new neg. log-likelihood / old neg. log-likelihood) | < tol

    verbose : int, optional, default: 0
        Specifies the amount of additional debug information
        during optimization.

    Attributes
    ----------
    coef_ : ndarray, shape = (n_features,)
        Coefficients of the model

    cum_baseline_hazard_ : ``sksurv.functions.StepFunction``
        Estimated baseline cumulative hazard function.

    baseline_survival_ : ``sksurv.functions.StepFunction``
        Estimated baseline survival function.

    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.
    """

    _tags = {"authors": ["sebp", "fkiraly"]}  # sebp credit for interfaced estimator

    def __init__(self, alpha=0, ties="breslow", n_iter=100, tol=1e-9, verbose=0):
        self.alpha = alpha
        self.ties = ties
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

        super().__init__()

    def _get_sksurv_class(self):
        """Getter of the sksurv class to be used for the adapter."""
        from sksurv.linear_model import CoxPHSurvivalAnalysis as _CoxPH

        return _CoxPH

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

        params2 = {
            "alpha": 0.1,
            "ties": "efron",
            "n_iter": 99,
            "tol": 1e-7,
            "verbose": 1,
        }

        return [params1, params2]
