"""Density-based outlier detection using probabilistic regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["DensityOutlierDetector"]

import numpy as np
import pandas as pd

from skpro.outlier.base import BaseOutlierDetector


class DensityOutlierDetector(BaseOutlierDetector):
    """Density-based outlier detection using probabilistic regressors.

    Detects outliers based on the probability density of the observed values
    under the predictive distribution. Samples with low density (high negative
    log-likelihood) are considered outliers.

    Parameters
    ----------
    regressor : skpro probabilistic regressor
        A fitted or unfitted probabilistic regressor implementing the
        skpro BaseProbaRegressor interface with predict_proba capability.
    contamination : float, default=0.1
        The proportion of outliers in the dataset.
    use_log : bool, default=True
        If True, use negative log-likelihood as the outlier score.
        If False, use negative likelihood directly.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
    threshold_ : float
        The threshold used to determine outliers.

    Examples
    --------
    >>> from skpro.regression.residual import ResidualDouble
    >>> from skpro.outlier import DensityOutlierDetector
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    >>> reg = ResidualDouble(RandomForestRegressor(random_state=42))
    >>> detector = DensityOutlierDetector(reg, contamination=0.1)
    >>> detector.fit(X, y)
    DensityOutlierDetector(...)
    >>> outliers = detector.predict(X, y)
    >>> int(outliers.sum())  # number of detected outliers
    10
    """

    _tags = {
        "object_type": "outlier_detector",
        "estimator_type": "outlier_detector",
        "authors": ["skpro developers"],
    }

    def __init__(self, regressor, contamination=0.1, use_log=True):
        self.use_log = use_log
        super().__init__(regressor=regressor, contamination=contamination)

    def _compute_decision_scores(self, X, y=None):
        """Compute density-based outlier scores.

        The score is computed as the negative log-pdf (or negative pdf) of
        the observed values under the predictive distribution. Higher scores
        indicate lower probability density, i.e., more outlier-like samples.

        Parameters
        ----------
        X : pandas DataFrame
            Feature data
        y : pandas DataFrame or pandas Series, default=None
            Target data. Required for computing outlier scores.

        Returns
        -------
        scores : numpy array of shape (n_samples,)
            Outlier scores for each sample
        """
        if y is None:
            raise ValueError(
                "Target variable y is required for density-based " "outlier detection."
            )

        # Ensure y is a Series or DataFrame
        if isinstance(y, pd.Series):
            y_df = y.to_frame()
        elif isinstance(y, pd.DataFrame):
            y_df = y
        else:
            y_df = pd.DataFrame(y)

        # Get predictive distribution
        y_pred_dist = self.regressor.predict_proba(X)

        # Compute log-pdf of observed values
        log_pdf = y_pred_dist.log_pdf(y_df)

        # Convert to numpy array
        if isinstance(log_pdf, (pd.DataFrame, pd.Series)):
            log_pdf = log_pdf.values

        # For multi-output, sum across outputs
        if log_pdf.ndim > 1:
            log_pdf = np.sum(log_pdf, axis=1)

        # Convert to outlier score
        if self.use_log:
            # Negative log-likelihood as score
            scores = -log_pdf
        else:
            # Negative likelihood as score
            scores = -np.exp(log_pdf)

        # Handle potential infinities or NaNs
        scores = np.nan_to_num(scores, nan=np.inf, posinf=np.inf, neginf=0.0)

        return scores

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
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        from skpro.regression.residual import ResidualDouble

        params1 = {"regressor": ResidualDouble(RandomForestRegressor(n_estimators=2))}
        params2 = {
            "regressor": ResidualDouble(LinearRegression()),
            "contamination": 0.05,
            "use_log": False,
        }
        return [params1, params2]
