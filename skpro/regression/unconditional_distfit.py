"""Unconditional probabilistic regression baseline using distfit.

Regressor ignores all features and fits a univariate density to target using distfit.
"""


import numpy as np

from skpro.distributions.base import BaseDistribution
from skpro.regression.base import BaseProbaRegressor


class UnconditionalDistfitRegressor(BaseProbaRegressor):
    """
    Featureless unconditional probabilistic regressor using distfit.

    Fits a univariate density to the target using distfit, ignoring all features.
    Supports parametric (e.g., normal, laplace) and histogram fitting via distfit's API.
    Multi-output y is not supported (raises NotImplementedError).

    This is a constant-uncertainty baseline: uncertainty does not shrink with more
    data. Requires the optional dependency `distfit` (install with
    `pip install distfit`).

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.regression.unconditional_distfit import (
    ...     UnconditionalDistfitRegressor
    ... )
    >>> y = pd.DataFrame([1, 2, 3, 4, 5])
    >>> X = pd.DataFrame(index=y.index)  # featureless DataFrame
    >>> reg = UnconditionalDistfitRegressor(distr_type="norm")
    >>> reg.fit(X, y)
    UnconditionalDistfitRegressor()
    >>> dist = reg.predict_proba(X)
    >>> float(dist.mean().iloc[0, 0])
    3.0

    References
    ----------
    - mlr3proba: Probabilistic Supervised Learning in R (density estimation).
      https://mlr3book.mlr-org.com/chapters/chapter13/beyond_regression_and_classification.html
    - LinCDE: Conditional Density Estimation via Lindsey’s Method
      (Gao & Hastie, JMLR 2022). https://jmlr.org/papers/volume23/21-0840/21-0840.pdf
    - Conditional Density Estimation with Histogram Trees (Yang et al., NeurIPS 2024).
      https://arxiv.org/html/2410.11449v1
    - Nonparametric Conditional Density Estimation (Hansen, 2004).
      https://users.ssc.wisc.edu/~behansen/papers/ncde.pdf
    - distfit documentation: https://erdogant.github.io/distfit/
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["arnavk23"],
        "estimator_type": "regressor_proba",
        "python_dependencies": "distfit>=1.6.8",
        # estimator tags
        # --------------
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        # CI and test flags
        # -----------------
        "tests:vm": False,  # set True if special VM is needed
    }

    def __init__(self, distr_type="norm", random_state=None, fit_histogram=False):
        """Initialize UnconditionalDistfitRegressor.

        Parameters
        ----------
        distr_type : str, default='norm'
            Distribution type for distfit (e.g., 'norm', 'laplace'; see distfit docs).
        random_state : int or None
            Random seed for reproducibility.
        fit_histogram : bool, default=False
            If True, fit a histogram using distfit's histogram option.
        """
        allowed_types = ["norm", "laplace", "histogram"]
        if distr_type not in allowed_types:
            raise ValueError(
                f"distr_type must be one of {allowed_types}, got {distr_type}"
            )
        self.distr_type = distr_type
        self.random_state = random_state
        self.fit_histogram = fit_histogram
        super().__init__()

    def _fit(self, X, y, C=None):
        # Import distfit only when needed for dependency isolation
        from distfit import distfit

        y_arr_raw = y.values if hasattr(y, "values") else np.asarray(y)
        if y_arr_raw.ndim > 2 or (y_arr_raw.ndim == 2 and y_arr_raw.shape[1] > 1):
            raise NotImplementedError(
                "UnconditionalDistfitRegressor only supports univariate y. Got shape: "
                + str(y_arr_raw.shape)
            )

        if hasattr(y, "columns"):
            self._y_cols = y.columns
        else:
            self._y_cols = ["0"]

        y_arr = np.asarray(y_arr_raw).reshape(-1)
        if self.distr_type == "kde":
            raise RuntimeError(
                "KDE support is removed due to scipy.stats.kde deprecation in distfit. "
                "Please use a different distribution type."
            )
        if self.fit_histogram:
            raise RuntimeError(
                "Histogram support is not available in distfit>=2.0.1. "
                "Please set fit_histogram=False and use a parametric distr_type "
                "such as 'norm' or 'laplace'."
            )
        else:
            self.distfit_ = distfit(
                distr=self.distr_type, random_state=self.random_state
            )
        self.distfit_.fit_transform(y_arr)
        return self

    def _predict_proba(self, X):
        # Return one-row-per-instance distribution with y-aligned columns.
        return _DistfitDistribution(self.distfit_, index=X.index, columns=self._y_cols)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for automated tests."""
        return [
            {"distr_type": "norm", "fit_histogram": False},
            {"distr_type": "laplace", "fit_histogram": False},
        ]


class _DistfitDistribution(BaseDistribution):
    """Wraps a distfit fitted object as a skpro distribution."""

    def __init__(self, distfit_obj, index=None, columns=None, distr_type=None):
        if isinstance(distfit_obj, np.ndarray):
            distfit_obj = distfit_obj.item()

        self.distfit_obj = distfit_obj
        if distr_type is None:
            distr_type = getattr(self.distfit_obj, "distr", None)
        self.distr_type = distr_type
        super().__init__(index=index, columns=columns)

    def _get_fitted_model(self):
        """Return fitted scipy frozen distribution when available."""
        model = self.distfit_obj.model
        if isinstance(model, dict):
            model = model.get("model", model)
        return model

    def _get_scalar_mean(self):
        """Return scalar mean for the fitted distribution."""
        model = self.distfit_obj.model
        if isinstance(model, dict):
            if "loc" in model:
                return float(model["loc"])
            if "mean" in model:
                return float(model["mean"])
        fitted = self._get_fitted_model()
        return float(fitted.mean())

    def _get_scalar_var(self):
        """Return scalar variance for the fitted distribution."""
        model = self.distfit_obj.model
        if isinstance(model, dict) and "scale" in model:
            return float(model["scale"]) ** 2
        fitted = self._get_fitted_model()
        return float(fitted.var())

    def _mean(self):
        return np.full(self.shape, self._get_scalar_mean(), dtype=float)

    def _var(self):
        return np.full(self.shape, self._get_scalar_var(), dtype=float)

    def _pdf(self, x):
        fitted = self._get_fitted_model()
        return fitted.pdf(x)

    def _cdf(self, x):
        fitted = self._get_fitted_model()
        return fitted.cdf(x)

    def _ppf(self, p):
        fitted = self._get_fitted_model()
        return fitted.ppf(p)

    def get_params(self, deep=True):
        """Return parameters of the distribution."""
        # Example: expose distfit_obj and its distribution type if available
        distr_type = getattr(self.distfit_obj, "distr", None)
        return {"distfit_obj": self.distfit_obj, "distr_type": distr_type}
