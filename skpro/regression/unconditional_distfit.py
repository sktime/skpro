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
    Supports parametric (e.g., normal, laplace), nonparametric (kde),
    and histogram fitting via distfit's API.

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
        """
        Initialize UnconditionalDistfitRegressor.

        Parameters
        ----------
        distr_type : str, default='norm'
            Distribution type for distfit (e.g., 'norm', 'laplace', etc.; see
            distfit docs for full list).
        random_state : int or None
            Random seed for reproducibility.
        fit_histogram : bool, default=False
            If True, fit a histogram using distfit's histogram option.
        """
        self.distr_type = distr_type
        self.random_state = random_state
        self.fit_histogram = fit_histogram
        super().__init__()

    def _fit(self, X, y, C=None):
        # Import distfit only when needed for dependency isolation
        from distfit import distfit

        y_arr = y.values.flatten() if hasattr(y, "values") else np.asarray(y).flatten()
        # KDE support removed due to scipy.stats.kde deprecation in distfit
        if self.distr_type == "kde":
            raise RuntimeError(
                "KDE support is removed due to scipy.stats.kde deprecation in distfit. "
                "Please use a different distribution type."
            )
        if self.fit_histogram:
            self.distfit_ = distfit(distr="histogram", random_state=self.random_state)
        else:
            self.distfit_ = distfit(
                distr=self.distr_type, random_state=self.random_state
            )
        self.distfit_.fit_transform(y_arr)
        return self

    def _predict_proba(self, X):
        # Return a single distribution object for all samples
        return _DistfitDistribution(self.distfit_)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for automated tests."""
        return [
            {"distr_type": "norm", "fit_histogram": False},
            {"distr_type": "laplace", "fit_histogram": False},
        ]


class _DistfitDistribution(BaseDistribution):
    """Wraps a distfit fitted object as a skpro distribution."""

    def __init__(self, distfit_obj):
        self.distfit_obj = distfit_obj
        super().__init__()

    def sample(self, n_samples=1):
        return self.distfit_obj.generate(n_samples)

    def pdf(self, x):
        return self.distfit_obj.model.pdf(x)

    def mean(self):
        model = self.distfit_obj.model
        if isinstance(model, dict):
            # distfit returns 'loc' for normal/laplace, sometimes 'mean' for others
            if "loc" in model:
                return model["loc"]
            if "mean" in model:
                return model["mean"]
            raise AttributeError(
                "distfit dict has neither 'loc' nor 'mean' key; cannot determine mean."
            )
        return model.mean()

    def var(self):
        # For normal/laplace, variance is scale**2
        model = self.distfit_obj.model
        if isinstance(model, dict) and "scale" in model:
            return model["scale"] ** 2
        raise AttributeError(
            "distfit model does not have a 'scale' (variance) attribute"
        )

    def get_params(self, deep=True):
        """Return parameters of the distribution."""
        # Example: expose distfit_obj and its distribution type if available
        distr_type = getattr(self.distfit_obj, "distr", None)
        return {"distfit_obj": self.distfit_obj, "distr_type": distr_type}
