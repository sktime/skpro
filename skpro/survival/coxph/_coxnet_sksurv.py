"""Interface adapters to scikit-survival Cox-net model."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from skpro.survival.adapters.sksurv import _SksurvAdapter
from skpro.survival.base import BaseSurvReg


class CoxNet(_SksurvAdapter, BaseSurvReg):
    """Cox proportional hazards model with elastic net penalty.

    Direct interface to ``sksurv.linear_model.CoxnetSurvivalAnalysis``, by ``sebp``.

    Parameters
    ----------
    n_alphas : int, optional, default: 100
        Number of alphas along the regularization path.

    alphas : array-like or None, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically.

    alpha_min_ratio : float or { "auto" }, optional, default: "auto"
        Determines minimum alpha of the regularization path
        if ``alphas`` is ``None``. The smallest value for alpha
        is computed as the fraction of the data derived maximum
        alpha (i.e. the smallest value for which all
        coefficients are zero).

        If set to "auto", the value will depend on the
        sample size relative to the number of features.
        If ``n_samples > n_features``, the default value is 0.0001
        If ``n_samples <= n_features``, 0.01 is the default value.

    l1_ratio : float, optional, default: 0.5
        The ElasticNet mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty.
        For ``l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

    penalty_factor : array-like or None, optional
        Separate penalty factors can be applied to each coefficient.
        This is a number that multiplies alpha to allow differential
        shrinkage.  Can be 0 for some variables, which implies no shrinkage,
        and that variable is always included in the model.
        Default is 1 for all variables.

        Note: the penalty factors are internally rescaled to sum to
        ``n_features``, and the alphas sequence will reflect this change.

    normalize : boolean, optional, default: False
        If True, the features X will be normalized before optimization by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        ``sklearn.preprocessing.StandardScaler`` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : boolean, optional, default: True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, optional, default: 1e-7
        The tolerance for the optimization: optimization continues
        until all updates are smaller than ``tol``.

    max_iter : int, optional, default: 100000
        The maximum number of iterations.

    verbose : bool, optional, default: False
        Whether to print additional information during optimization.

    Attributes
    ----------
    alphas_ : ndarray, shape=(n_alphas,)
        The actual sequence of alpha values used.

    alpha_min_ratio_ : float
        The inferred value of alpha_min_ratio.

    penalty_factor_ : ndarray, shape=(n_features,)
        The actual penalty factors used.

    coef_ : ndarray, shape=(n_features, n_alphas)
        Matrix of coefficients.

    offset_ : ndarray, shape=(n_alphas,)
        Bias term to account for non-centered features.

    deviance_ratio_ : ndarray, shape=(n_alphas,)
        The fraction of (null) deviance explained.

    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.

    Examples
    --------
    >>> from skpro.survival.linear import CoxNet  # doctest: +SKIP
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>> from sklearn.model_selection import train_test_split  # doctest: +SKIP
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)  # doctest: +SKIP

    >>> reg_proba = CoxNet()  # doctest: +SKIP
    >>> reg_proba.fit(X_train, y_train)  # doctest: +SKIP
    >>> y_pred = reg_proba.predict_proba(X_test)  # doctest: +SKIP
    """

    _tags = {"authors": ["sebp", "fkiraly"]}  # sebp credit for interfaced estimator

    def __init__(
        self,
        n_alphas=100,
        alphas=None,
        alpha_min_ratio="auto",
        l1_ratio=0.5,
        penalty_factor=None,
        normalize=False,
        copy_X=True,
        tol=1e-7,
        max_iter=100000,
        verbose=False,
    ):
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.l1_ratio = l1_ratio
        self.penalty_factor = penalty_factor
        self.normalize = normalize
        self.copy_X = copy_X
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

        super().__init__()

    def _get_sksurv_class(self):
        """Getter of the sksurv class to be used for the adapter."""
        from sksurv.linear_model import CoxnetSurvivalAnalysis as _CoxNet

        return _CoxNet

    def _get_sksurv_object(self):
        """Abstract method to initialize sksurv object.

        The default initializes result of _get_sksurv_class
        with self.get_params.
        """
        cls = self._get_sksurv_class()
        params = self.get_params()
        params["fit_baseline_model"] = True  # required for predict_survival_function
        # and therefore for _predict_proba implementation to be valid
        return cls(**params)

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
            "n_alphas": 99,
            "alpha_min_ratio": 0.001,
            "l1_ratio": 0.4,
            "normalize": True,
            "tol": 1e-6,
            "max_iter": 99999,
            "verbose": True,
        }

        return [params1, params2]
