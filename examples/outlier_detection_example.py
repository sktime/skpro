"""Example demonstrating outlier detection with probabilistic regressors.

This example shows how to use the three types of outlier detectors:
1. QuantileOutlierDetector - based on predictive quantiles
2. DensityOutlierDetector - based on probability density
3. LossOutlierDetector - based on predictive loss

All three implement a pyod-compatible interface.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from skpro.outlier import (
    DensityOutlierDetector,
    LossOutlierDetector,
    QuantileOutlierDetector,
)
from skpro.regression.residual import ResidualDouble

# %%
# Create synthetic dataset with outliers
# ---------------------------------------
# We'll create a simple regression dataset and inject some outliers

np.random.seed(42)
n_samples = 200
n_outliers = 20

# Generate normal data
X = np.random.randn(n_samples, 5)
y = (
    3 * X[:, 0]
    + 2 * X[:, 1]
    - X[:, 2]
    + 0.5 * X[:, 3]
    + np.random.randn(n_samples) * 0.5
)

# Add outliers with larger residuals
outlier_indices = np.random.choice(n_samples, size=n_outliers, replace=False)
y[outlier_indices] += np.random.randn(n_outliers) * 5

X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
y_series = pd.Series(y, name="y")

print(f"Dataset created: {n_samples} samples with {n_outliers} outliers")
print(f"True outlier indices: {sorted(outlier_indices)[:10]}...")

# %%
# Example 1: Quantile-based Outlier Detection
# --------------------------------------------
# Detects outliers based on whether observations fall outside predicted quantile ranges

regressor1 = ResidualDouble(RandomForestRegressor(n_estimators=100, random_state=42))
detector1 = QuantileOutlierDetector(
    regressor1, contamination=n_outliers / n_samples, alpha=[0.05, 0.95]
)

detector1.fit(X_df, y_series)
outliers1 = detector1.predict(X_df, y_series)
scores1 = detector1.decision_function(X_df, y_series)

print("\nQuantileOutlierDetector:")
print(f"  Number of outliers detected: {np.sum(outliers1)}")
print(f"  Detected outlier indices: {np.where(outliers1 == 1)[0][:10]}...")
print(f"  Threshold: {detector1.threshold_:.4f}")

# Calculate precision and recall
true_positives = np.sum(outliers1[outlier_indices])
detected_outliers = np.sum(outliers1)
precision = true_positives / detected_outliers if detected_outliers > 0 else 0
recall = true_positives / n_outliers
print(f"  Precision: {precision:.2%}, Recall: {recall:.2%}")

# %%
# Example 2: Density-based Outlier Detection
# -------------------------------------------
# Detects outliers based on low probability density (high negative log-likelihood)

regressor2 = ResidualDouble(LinearRegression())
detector2 = DensityOutlierDetector(
    regressor2, contamination=n_outliers / n_samples, use_log=True
)

detector2.fit(X_df, y_series)
outliers2 = detector2.predict(X_df, y_series)
scores2 = detector2.decision_function(X_df, y_series)

print("\nDensityOutlierDetector:")
print(f"  Number of outliers detected: {np.sum(outliers2)}")
print(f"  Detected outlier indices: {np.where(outliers2 == 1)[0][:10]}...")
print(f"  Threshold: {detector2.threshold_:.4f}")

# Calculate precision and recall
true_positives2 = np.sum(outliers2[outlier_indices])
detected_outliers2 = np.sum(outliers2)
precision2 = true_positives2 / detected_outliers2 if detected_outliers2 > 0 else 0
recall2 = true_positives2 / n_outliers
print(f"  Precision: {precision2:.2%}, Recall: {recall2:.2%}")

# %%
# Example 3: Loss-based Outlier Detection
# ----------------------------------------
# Detects outliers based on predictive loss (e.g., log-loss, CRPS, interval score)

regressor3 = ResidualDouble(RandomForestRegressor(n_estimators=100, random_state=42))

# Test with different loss functions
loss_functions = ["log_loss", "crps", "interval_score"]

for loss_fn in loss_functions:
    detector3 = LossOutlierDetector(
        ResidualDouble(RandomForestRegressor(n_estimators=100, random_state=42)),
        contamination=n_outliers / n_samples,
        loss=loss_fn,
    )

    detector3.fit(X_df, y_series)
    outliers3 = detector3.predict(X_df, y_series)
    scores3 = detector3.decision_function(X_df, y_series)

    print(f"\nLossOutlierDetector (loss={loss_fn}):")
    print(f"  Number of outliers detected: {np.sum(outliers3)}")
    print(f"  Detected outlier indices: {np.where(outliers3 == 1)[0][:10]}...")
    print(f"  Threshold: {detector3.threshold_:.4f}")

    # Calculate precision and recall
    true_positives3 = np.sum(outliers3[outlier_indices])
    detected_outliers3 = np.sum(outliers3)
    precision3 = true_positives3 / detected_outliers3 if detected_outliers3 > 0 else 0
    recall3 = true_positives3 / n_outliers
    print(f"  Precision: {precision3:.2%}, Recall: {recall3:.2%}")

# %%
# Example 4: Custom Loss Function
# --------------------------------
# You can also define your own loss function


def custom_loss(y_true, y_pred_dist):
    """Compute squared error loss."""
    y_pred = y_pred_dist.mean()
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.values
    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true = y_true.values
    return (y_true.flatten() - y_pred.flatten()) ** 2


detector4 = LossOutlierDetector(
    ResidualDouble(LinearRegression()),
    contamination=n_outliers / n_samples,
    loss=custom_loss,
)

detector4.fit(X_df, y_series)
outliers4 = detector4.predict(X_df, y_series)

print("\nLossOutlierDetector (custom loss - squared error):")
print(f"  Number of outliers detected: {np.sum(outliers4)}")
print(f"  Detected outlier indices: {np.where(outliers4 == 1)[0][:10]}...")

# Calculate precision and recall
true_positives4 = np.sum(outliers4[outlier_indices])
detected_outliers4 = np.sum(outliers4)
precision4 = true_positives4 / detected_outliers4 if detected_outliers4 > 0 else 0
recall4 = true_positives4 / n_outliers
print(f"  Precision: {precision4:.2%}, Recall: {recall4:.2%}")

# %%
# Visualization (optional - requires matplotlib)
# -----------------------------------------------
try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Outlier scores from QuantileOutlierDetector
    ax = axes[0, 0]
    ax.scatter(
        range(n_samples),
        scores1,
        c=outliers1,
        cmap="RdYlGn_r",
        alpha=0.6,
        edgecolors="k",
    )
    ax.axhline(y=detector1.threshold_, color="r", linestyle="--", label="Threshold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Outlier Score")
    ax.set_title("QuantileOutlierDetector Scores")
    ax.legend()

    # Plot 2: Outlier scores from DensityOutlierDetector
    ax = axes[0, 1]
    ax.scatter(
        range(n_samples),
        scores2,
        c=outliers2,
        cmap="RdYlGn_r",
        alpha=0.6,
        edgecolors="k",
    )
    ax.axhline(y=detector2.threshold_, color="r", linestyle="--", label="Threshold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Outlier Score")
    ax.set_title("DensityOutlierDetector Scores")
    ax.legend()

    # Plot 3: Feature space visualization (first 2 features)
    ax = axes[1, 0]
    inliers = outliers1 == 0
    outliers_mask = outliers1 == 1
    ax.scatter(
        X_df.iloc[inliers, 0],
        X_df.iloc[inliers, 1],
        c="blue",
        label="Inliers",
        alpha=0.6,
    )
    ax.scatter(
        X_df.iloc[outliers_mask, 0],
        X_df.iloc[outliers_mask, 1],
        c="red",
        label="Outliers",
        alpha=0.8,
        marker="x",
        s=100,
    )
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title("Outliers in Feature Space (QuantileOutlierDetector)")
    ax.legend()

    # Plot 4: Comparison of methods
    ax = axes[1, 1]
    methods = ["Quantile", "Density", "Loss (log)"]
    recalls = [recall, recall2, recall3]
    precisions = [precision, precision2, precision3]

    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width / 2, recalls, width, label="Recall", alpha=0.8)
    ax.bar(x + width / 2, precisions, width, label="Precision", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Comparison of Detection Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig("outlier_detection_example.png", dpi=150)
    print("\nVisualization saved as 'outlier_detection_example.png'")
    plt.show()

except ImportError:
    print("\nSkipping visualization (matplotlib not installed)")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)
