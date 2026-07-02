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
    output_agg : {"sum", "mean"}, default="sum"
        Aggregation used when a loss returns multi-output scores of shape
        ``(n_samples, n_outputs)``. Applied consistently to built-in and
        callable losses.

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

    def __init__(
        self,
        regressor,
        contamination=0.1,
        loss="log_loss",
        alpha=0.05,
        output_agg="sum",
    ):
        if output_agg not in {"sum", "mean"}:
            raise ValueError(
                f"Invalid output_agg={output_agg!r}. Must be 'sum' or 'mean'."
            )
        self.loss = loss
        self.alpha = alpha
        self.output_agg = output_agg
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

        # Get predictive distribution from fitted clone if available,
        # otherwise from the user-passed regressor (for direct unit-test calls).
        regressor = self.regressor_ if hasattr(self, "regressor_") else self.regressor
        y_pred_dist = regressor.predict_proba(X)

        # Compute loss based on specified loss type
        if callable(self.loss):
            # Custom loss function
            scores = self.loss(y_df, y_pred_dist)
        elif self.loss == "log_loss":
            # Negative log-likelihood
            log_pdf = y_pred_dist.log_pdf(y_df)
            if isinstance(log_pdf, (pd.DataFrame, pd.Series)):
                log_pdf = log_pdf.values
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

        # Ensure scores are a 1D array of shape (n_samples,)
        scores = np.asarray(scores)
        n_samples = len(y_df)

        if scores.ndim == 2:
            if scores.shape[0] != n_samples:
                raise ValueError(
                    "Loss function must return scores aligned with the number "
                    f"of samples. Expected first dimension {n_samples}, got "
                    f"array with shape {scores.shape!r}."
                )
            scores = self._aggregate_multioutput_scores(scores)
        elif scores.ndim != 1:
            raise ValueError(
                "Loss function must return scores of shape (n_samples,) or "
                "(n_samples, n_outputs). Got array with shape "
                f"{scores.shape!r}."
            )

        if scores.shape[0] != n_samples:
            raise ValueError(
                "Loss function must return one score per sample. "
                f"Expected shape ({n_samples},) after reduction, got "
                f"{scores.shape!r}."
            )

        return scores

    def _aggregate_multioutput_scores(self, scores):
        """Aggregate multi-output scores to one score per sample."""
        if scores.shape[1] == 1:
            return scores.ravel()

        if self.output_agg == "sum":
            return np.sum(scores, axis=1)

        return np.mean(scores, axis=1)

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
            "loss": "log_loss",
        }
        params3 = {
            "regressor": ResidualDouble(RandomForestRegressor(n_estimators=2)),
            "loss": "crps",
        }
        params4 = {
            "regressor": ResidualDouble(RandomForestRegressor(n_estimators=2)),
            "loss": "interval_score",
            "output_agg": "mean",
        }
        return [params1, params2, params3, params4]

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
            crps = np.abs(crps)

        if isinstance(crps, (pd.DataFrame, pd.Series)):
            crps = crps.values

        crps = np.asarray(crps)

        if crps.ndim == 2 and crps.shape[1] == 1:
            crps = crps.ravel()
        elif crps.ndim > 1:
            crps = self._aggregate_multioutput_scores(crps)

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
        regressor = self.regressor_ if hasattr(self, "regressor_") else self.regressor
        intervals = regressor.predict_interval(X, coverage=coverage)

        lower, upper = self._extract_interval_bounds(intervals)

        # Convert y_true to array
        if isinstance(y_true, (pd.DataFrame, pd.Series)):
            y_true_arr = y_true.values
        else:
            y_true_arr = y_true

        if y_true_arr.ndim == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)

        if lower.ndim == 1:
            lower = lower.reshape(-1, 1)
        if upper.ndim == 1:
            upper = upper.reshape(-1, 1)

        if (
            lower.shape[0] != y_true_arr.shape[0]
            or upper.shape[0] != y_true_arr.shape[0]
            or lower.shape[1] != y_true_arr.shape[1]
            or upper.shape[1] != y_true_arr.shape[1]
        ):
            raise ValueError(
                "predict_interval output is not aligned with y_true shape. "
                f"Expected ({y_true_arr.shape[0]}, {y_true_arr.shape[1]}) for "
                f"lower/upper, got lower={lower.shape!r}, upper={upper.shape!r}."
            )

        # Compute interval score
        # IS = (upper - lower) + (2/alpha) * (lower - y) * I(y < lower)
        #                       + (2/alpha) * (y - upper) * I(y > upper)
        width = upper - lower
        violation_lower = np.maximum(0, lower - y_true_arr) * (2 / self.alpha)
        violation_upper = np.maximum(0, y_true_arr - upper) * (2 / self.alpha)

        scores = width + violation_lower + violation_upper

        return scores

    def _extract_interval_bounds(self, intervals):
        """Extract lower/upper interval bounds from common predict_interval formats."""
        if isinstance(intervals, pd.DataFrame):
            cols = intervals.columns

            if isinstance(cols, pd.MultiIndex):
                last_level = cols.get_level_values(-1)
                if "lower" in last_level and "upper" in last_level:
                    lower = intervals.xs("lower", axis=1, level=-1).to_numpy()
                    upper = intervals.xs("upper", axis=1, level=-1).to_numpy()
                    return lower, upper

            lower_cols = [col for col in cols if "lower" in str(col).lower()]
            upper_cols = [col for col in cols if "upper" in str(col).lower()]
            if lower_cols and upper_cols:
                return (
                    intervals[lower_cols].to_numpy(),
                    intervals[upper_cols].to_numpy(),
                )

            n_cols = len(cols)
            return (
                intervals.iloc[:, : n_cols // 2].to_numpy(),
                intervals.iloc[:, n_cols // 2 :].to_numpy(),
            )

        intervals = np.asarray(intervals)
        if intervals.ndim == 2:
            return intervals[:, 0], intervals[:, 1]
        if intervals.ndim == 3:
            return intervals[:, :, 0], intervals[:, :, 1]

        raise ValueError(
            "Unsupported predict_interval output format. "
            f"Got array-like with shape {intervals.shape!r}."
        )
