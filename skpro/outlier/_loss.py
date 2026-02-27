"""Loss-based outlier detection using probabilistic regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["LossOutlierDetector"]

import numpy as np
import pandas as pd

from skpro.outlier.base import BaseOutlierDetector


class LossOutlierDetector(BaseOutlierDetector):
    """Loss-based outlier detection using probabilistic regressors.

    Detects outliers based on the predictive loss. Samples with high
    predictive loss are considered outliers. The loss can be any metric
    that evaluates the quality of probabilistic predictions.

    Parameters
    ----------
    regressor : skpro probabilistic regressor
        A fitted or unfitted probabilistic regressor implementing the
        skpro BaseProbaRegressor interface.
    contamination : float, default=0.1
        The proportion of outliers in the dataset.
    loss : str or callable, default="log_loss"
        The loss function to use. Can be:
        - "log_loss": negative log-likelihood (same as density-based)
        - "crps": Continuous Ranked Probability Score
        - "interval_score": interval score for a given coverage
        - callable: custom loss function that takes (y_true, y_pred_dist)
          and returns array of losses
    alpha : float, default=0.05
        Significance level for interval-based losses.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
    threshold_ : float
        The threshold used to determine outliers.

    Examples
    --------
    >>> from skpro.regression.residual import ResidualDouble
    >>> from skpro.outlier import LossOutlierDetector
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    >>> reg = ResidualDouble(RandomForestRegressor(random_state=42))
    >>> detector = LossOutlierDetector(reg, contamination=0.1, loss="log_loss")
    >>> detector.fit(X, y)
    LossOutlierDetector(...)
    >>> outliers = detector.predict(X, y)
    >>> int(outliers.sum())  # number of detected outliers
    10
    """

    _tags = {
        "object_type": "outlier_detector",
        "estimator_type": "outlier_detector",
        "authors": ["skpro developers"],
    }

    def __init__(self, regressor, contamination=0.1, loss="log_loss", alpha=0.05):
        self.loss = loss
        self.alpha = alpha
        super().__init__(regressor=regressor, contamination=contamination)

    def _compute_decision_scores(self, X, y=None):
        """Compute loss-based outlier scores.

        The score is computed using the specified loss function on the
        predictive distribution. Higher losses indicate worse predictions,
        i.e., more outlier-like samples.

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
                "Target variable y is required for loss-based " "outlier detection."
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

        # Compute loss based on specified loss type
        if callable(self.loss):
            # Custom loss function
            scores = self.loss(y_df, y_pred_dist)
        elif self.loss == "log_loss":
            # Negative log-likelihood
            log_pdf = y_pred_dist.log_pdf(y_df)
            if isinstance(log_pdf, (pd.DataFrame, pd.Series)):
                log_pdf = log_pdf.values
            if log_pdf.ndim > 1:
                log_pdf = np.sum(log_pdf, axis=1)
            scores = -log_pdf
        elif self.loss == "crps":
            # CRPS (Continuous Ranked Probability Score)
            scores = self._compute_crps(y_df, y_pred_dist)
        elif self.loss == "interval_score":
            # Interval score
            scores = self._compute_interval_score(y_df, y_pred_dist, X)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss}. "
                "Must be 'log_loss', 'crps', 'interval_score', or a callable."
            )

        # Convert to numpy array if needed
        if isinstance(scores, (pd.DataFrame, pd.Series)):
            scores = scores.values

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
        from skpro.regression.residual import ResidualDouble
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        params1 = {"regressor": ResidualDouble(RandomForestRegressor(n_estimators=2))}
        params2 = {
            "regressor": ResidualDouble(LinearRegression()),
            "contamination": 0.05,
            "loss": "log_loss",
        }
        params3 = {
            "regressor": ResidualDouble(RandomForestRegressor(n_estimators=2)),
            "loss": "crps",
        }
        return [params1, params2, params3]

    def _compute_crps(self, y_true, y_pred_dist):
        """Compute Continuous Ranked Probability Score.

        CRPS measures the difference between the predicted distribution
        and the observed value.

        Parameters
        ----------
        y_true : pandas DataFrame
            True values
        y_pred_dist : skpro distribution
            Predicted distribution

        Returns
        -------
        crps : numpy array
            CRPS for each sample
        """
        # Try to get CRPS from distribution if available
        if hasattr(y_pred_dist, "crps"):
            crps = y_pred_dist.crps(y_true)
        else:
            # Approximate CRPS using mean absolute deviation
            # For a Normal distribution, CRPS ≈ σ * (1/√π - 2φ(z) + z(2Φ(z) - 1))
            # where z = (y - μ) / σ
            #
            # For simplicity, we use: CRPS ≈ E|Y - y|
            # where Y ~ predicted distribution

            # Use expected value and standard deviation
            y_pred_mean = y_pred_dist.mean()
            if isinstance(y_pred_mean, (pd.DataFrame, pd.Series)):
                y_pred_mean = y_pred_mean.values
            if y_pred_mean.ndim == 1:
                y_pred_mean = y_pred_mean.reshape(-1, 1)

            # Get standard deviation if available
            if hasattr(y_pred_dist, "std"):
                y_pred_std = y_pred_dist.std()
                if isinstance(y_pred_std, (pd.DataFrame, pd.Series)):
                    y_pred_std = y_pred_std.values
                if y_pred_std.ndim == 1:
                    y_pred_std = y_pred_std.reshape(-1, 1)
            else:
                y_pred_std = np.ones_like(y_pred_mean)

            # Convert y_true to array
            if isinstance(y_true, (pd.DataFrame, pd.Series)):
                y_true_arr = y_true.values
            else:
                y_true_arr = y_true

            if y_true_arr.ndim == 1:
                y_true_arr = y_true_arr.reshape(-1, 1)

            # Compute normalized residual
            z = (y_true_arr - y_pred_mean) / np.maximum(y_pred_std, 1e-10)

            # Approximate CRPS for Normal distribution
            # CRPS = σ * (z * (2Φ(z) - 1) + 2φ(z) - 1/√π)
            from scipy.stats import norm

            phi = norm.cdf(z)  # CDF
            pdf = norm.pdf(z)  # PDF

            crps = y_pred_std * (z * (2 * phi - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
            crps = np.abs(crps.flatten())

        if isinstance(crps, (pd.DataFrame, pd.Series)):
            crps = crps.values

        if crps.ndim > 1:
            crps = np.sum(crps, axis=1)

        return crps

    def _compute_interval_score(self, y_true, y_pred_dist, X):
        """Compute interval score.

        The interval score evaluates the quality of prediction intervals.
        It penalizes both the width of the interval and violations.

        Parameters
        ----------
        y_true : pandas DataFrame
            True values
        y_pred_dist : skpro distribution
            Predicted distribution
        X : pandas DataFrame
            Feature data (used for predict_interval)

        Returns
        -------
        scores : numpy array
            Interval scores for each sample
        """
        # Get prediction interval
        coverage = 1 - self.alpha
        intervals = self.regressor.predict_interval(X, coverage=coverage)

        # Extract lower and upper bounds
        if isinstance(intervals, pd.DataFrame):
            # Assuming columns are like ('lower', 0) and ('upper', 0)
            lower_cols = [
                col for col in intervals.columns if "lower" in str(col).lower()
            ]
            upper_cols = [
                col for col in intervals.columns if "upper" in str(col).lower()
            ]

            if len(lower_cols) == 0:
                # Alternative: first half is lower, second half is upper
                n_cols = len(intervals.columns)
                lower = intervals.iloc[:, : n_cols // 2].values
                upper = intervals.iloc[:, n_cols // 2 :].values
            else:
                lower = intervals[lower_cols].values
                upper = intervals[upper_cols].values
        else:
            lower = intervals[:, 0]
            upper = intervals[:, 1]

        # Convert y_true to array
        if isinstance(y_true, (pd.DataFrame, pd.Series)):
            y_true_arr = y_true.values
        else:
            y_true_arr = y_true

        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)

        # Compute interval score
        # IS = (upper - lower) + (2/alpha) * (lower - y) * I(y < lower)
        #                       + (2/alpha) * (y - upper) * I(y > upper)
        width = upper - lower
        violation_lower = np.maximum(0, lower - y_true_arr) * (2 / self.alpha)
        violation_upper = np.maximum(0, y_true_arr - upper) * (2 / self.alpha)

        scores = width + violation_lower + violation_upper

        # Sum over outputs if multi-output
        if scores.ndim > 1:
            scores = np.sum(scores, axis=1)

        return scores
