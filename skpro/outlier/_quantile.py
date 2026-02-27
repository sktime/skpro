"""Quantile-based outlier detection using probabilistic regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["QuantileOutlierDetector"]

import numpy as np
import pandas as pd

from skpro.outlier.base import BaseOutlierDetector


class QuantileOutlierDetector(BaseOutlierDetector):
    """Quantile-based outlier detection using probabilistic regressors.

    Detects outliers based on the extremity of predictive quantiles. Samples
    that fall outside the expected quantile range are considered outliers.
    The outlier score is computed as the distance from the median quantile,
    normalized by the quantile range.

    Parameters
    ----------
    regressor : skpro probabilistic regressor
        A fitted or unfitted probabilistic regressor implementing the
        skpro BaseProbaRegressor interface with predict_quantiles capability.
    contamination : float, default=0.1
        The proportion of outliers in the dataset.
    alpha : list of float, default=[0.05, 0.95]
        The quantile levels to use for outlier detection. The default
        [0.05, 0.95] creates a 90% prediction interval.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
    threshold_ : float
        The threshold used to determine outliers.

    Examples
    --------
    >>> from skpro.regression.residual import ResidualDouble
    >>> from skpro.outlier import QuantileOutlierDetector
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    >>> reg = ResidualDouble(RandomForestRegressor(random_state=42))
    >>> detector = QuantileOutlierDetector(reg, contamination=0.1)
    >>> detector.fit(X, y)
    QuantileOutlierDetector(...)
    >>> outliers = detector.predict(X, y)
    >>> outliers.sum()  # number of detected outliers
    10
    """

    _tags = {
        "object_type": "outlier_detector",
        "estimator_type": "outlier_detector",
        "authors": ["skpro developers"],
    }

    def __init__(self, regressor, contamination=0.1, alpha=None):
        self.alpha = alpha if alpha is not None else [0.05, 0.95]
        super().__init__(regressor=regressor, contamination=contamination)

    def _compute_decision_scores(self, X, y=None):
        """Compute quantile-based outlier scores.

        The score is computed as the distance of the observed value from
        the median prediction, normalized by the quantile range. Higher
        scores indicate samples that are further from the expected range.

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
                "Target variable y is required for quantile-based " "outlier detection."
            )

        # Ensure y is a Series or DataFrame
        if isinstance(y, pd.Series):
            y_arr = y.values.reshape(-1, 1)
        elif isinstance(y, pd.DataFrame):
            y_arr = y.values
        else:
            y_arr = np.array(y).reshape(-1, 1)

        # Get quantile predictions
        quantiles = sorted(self.alpha)
        q_pred = self.regressor.predict_quantiles(X, alpha=quantiles)

        # Convert to numpy array
        if isinstance(q_pred, pd.DataFrame):
            q_pred_arr = q_pred.values
        else:
            q_pred_arr = np.array(q_pred)

        # Compute scores based on quantile extremity
        # Score is the distance from median, normalized by quantile range
        n_samples = len(X)
        scores = np.zeros(n_samples)

        # Get the lower and upper quantile predictions
        # Reshape q_pred_arr to have shape (n_samples, n_outputs, n_quantiles)
        if len(q_pred_arr.shape) == 2:
            # Assume shape is (n_samples, n_quantiles) for single output
            q_lower = q_pred_arr[:, 0:1]  # shape (n_samples, 1)
            q_upper = q_pred_arr[:, -1:]  # shape (n_samples, 1)
        else:
            # Multi-output case
            q_lower = q_pred_arr[:, :, 0]  # shape (n_samples, n_outputs)
            q_upper = q_pred_arr[:, :, -1]  # shape (n_samples, n_outputs)

        # Compute the range
        q_range = q_upper - q_lower
        q_range = np.maximum(q_range, 1e-10)  # Avoid division by zero

        # Compute median (0.5 quantile)
        median_pred = self.regressor.predict(X)
        if isinstance(median_pred, (pd.DataFrame, pd.Series)):
            median_pred = median_pred.values
        if median_pred.ndim == 1:
            median_pred = median_pred.reshape(-1, 1)

        # Compute normalized distance from median
        distance_from_median = np.abs(y_arr - median_pred)

        # Check if outside quantile range
        below_lower = y_arr < q_lower
        above_upper = y_arr > q_upper

        # Score is distance from nearest quantile bound, normalized by range
        for i in range(n_samples):
            if below_lower[i].any():
                # Distance from lower quantile
                scores[i] = np.max((q_lower[i] - y_arr[i]) / q_range[i])
            elif above_upper[i].any():
                # Distance from upper quantile
                scores[i] = np.max((y_arr[i] - q_upper[i]) / q_range[i])
            else:
                # Inside quantile range - use normalized distance from median
                scores[i] = np.max(distance_from_median[i] / q_range[i])

        return scores
