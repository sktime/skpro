"""
models.py — Probabilistic Model Implementations
Supports: Bayesian Ridge, Gaussian Process, Quantile Regression, Gradient Boosting with intervals
"""

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Any


# ─── Available Models Registry ────────────────────────────────────────────────

MODELS = {
    "bayesian_ridge": {
        "name": "Bayesian Ridge Regression",
        "description": "A Bayesian approach to linear regression. Estimates both the regression coefficients and their uncertainty using prior distributions. Great for small-to-medium datasets.",
        "assumptions": "Linear relationship between features and target. Gaussian noise.",
        "strengths": "Fast, interpretable, provides natural uncertainty bounds.",
        "category": "Linear"
    },
    "gaussian_process": {
        "name": "Gaussian Process Regressor",
        "description": "Non-parametric probabilistic model that defines a distribution over functions. Naturally provides uncertainty estimates (predictive variance).",
        "assumptions": "Data can be described by a smooth function. Works best on small datasets (<1000 rows).",
        "strengths": "Very flexible, excellent uncertainty calibration, works well with small data.",
        "category": "Non-parametric"
    },
    "quantile_regression": {
        "name": "Quantile Regression",
        "description": "Fits three separate linear models for the median (50th), lower (10th), and upper (90th) quantiles. The spread gives the prediction interval.",
        "assumptions": "Linear relationship. Does not assume Gaussian errors.",
        "strengths": "Robust to outliers, no distributional assumptions, interval has clear probabilistic meaning.",
        "category": "Linear"
    },
    "gradient_boosting": {
        "name": "Gradient Boosting with Intervals",
        "description": "Uses three gradient boosting models trained on different quantiles (10th, 50th, 90th percentile) to produce predictions and uncertainty bands.",
        "assumptions": "None (non-parametric). Works well on complex tabular data.",
        "strengths": "High accuracy, handles non-linear patterns, robust prediction intervals.",
        "category": "Ensemble"
    },
}


def get_available_models():
    """Return list of model descriptors for the /models endpoint."""
    return [{"id": k, **v} for k, v in MODELS.items()]


def _compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


# ─── Individual Model Fitters ──────────────────────────────────────────────────

def _fit_bayesian_ridge(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = BayesianRidge(compute_score=True, max_iter=300)
    model.fit(X_train_s, y_train)

    y_pred, y_std = model.predict(X_test_s, return_std=True)

    # 90% prediction interval: mean ± 1.645 * std
    z = 1.645
    y_lower = y_pred - z * y_std
    y_upper = y_pred + z * y_std

    r2, mae, rmse = _compute_metrics(y_test, y_pred)

    return {
        "y_pred": y_pred,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "parameters": {
            "alpha_1": round(model.alpha_1, 6),
            "lambda_1": round(model.lambda_1, 6),
            "alpha_": round(float(model.alpha_), 4),
            "lambda_": round(float(model.lambda_), 4),
            "interval": "90% credible interval",
        }
    }


def _fit_gaussian_process(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Limit training data for GP (it's O(n^3) in time complexity)
    MAX_GP_SAMPLES = 300
    if len(X_train_s) > MAX_GP_SAMPLES:
        idx = np.random.choice(len(X_train_s), MAX_GP_SAMPLES, replace=False)
        X_train_s = X_train_s[idx]
        y_train = y_train[idx]

    # RBF kernel + white noise
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    model.fit(X_train_s, y_train)

    y_pred, y_std = model.predict(X_test_s, return_std=True)

    z = 1.645
    y_lower = y_pred - z * y_std
    y_upper = y_pred + z * y_std

    r2, mae, rmse = _compute_metrics(y_test, y_pred)

    return {
        "y_pred": y_pred,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "parameters": {
            "kernel": str(model.kernel_),
            "log_marginal_likelihood": round(float(model.log_marginal_likelihood_value_), 4),
            "interval": "90% predictive interval from GP posterior",
        }
    }


def _fit_quantile_regression(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Three quantile models
    model_median = QuantileRegressor(quantile=0.5, alpha=0.1, solver="highs")
    model_lower = QuantileRegressor(quantile=0.1, alpha=0.1, solver="highs")
    model_upper = QuantileRegressor(quantile=0.9, alpha=0.1, solver="highs")

    model_median.fit(X_train_s, y_train)
    model_lower.fit(X_train_s, y_train)
    model_upper.fit(X_train_s, y_train)

    y_pred = model_median.predict(X_test_s)
    y_lower = model_lower.predict(X_test_s)
    y_upper = model_upper.predict(X_test_s)

    # Ensure lower <= pred <= upper
    y_lower = np.minimum(y_lower, y_pred)
    y_upper = np.maximum(y_upper, y_pred)

    r2, mae, rmse = _compute_metrics(y_test, y_pred)

    return {
        "y_pred": y_pred,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "parameters": {
            "lower_quantile": "10th percentile",
            "median_quantile": "50th percentile",
            "upper_quantile": "90th percentile",
            "regularization_alpha": 0.1,
            "interval": "80% prediction interval (10th–90th percentile)",
        }
    }


def _fit_gradient_boosting(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    common_params = dict(n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_leaf=5)

    model_median = GradientBoostingRegressor(loss="squared_error", **common_params)
    model_lower = GradientBoostingRegressor(loss="quantile", alpha=0.1, **common_params)
    model_upper = GradientBoostingRegressor(loss="quantile", alpha=0.9, **common_params)

    model_median.fit(X_train_s, y_train)
    model_lower.fit(X_train_s, y_train)
    model_upper.fit(X_train_s, y_train)

    y_pred = model_median.predict(X_test_s)
    y_lower = model_lower.predict(X_test_s)
    y_upper = model_upper.predict(X_test_s)

    y_lower = np.minimum(y_lower, y_pred)
    y_upper = np.maximum(y_upper, y_pred)

    r2, mae, rmse = _compute_metrics(y_test, y_pred)

    return {
        "y_pred": y_pred,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "parameters": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
            "lower_quantile": "10th percentile",
            "upper_quantile": "90th percentile",
            "interval": "80% prediction interval (10th–90th percentile)",
        }
    }


# ─── Main Dispatcher ──────────────────────────────────────────────────────────

def fit_model(model_type: str, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Fit the requested model and return predictions + uncertainty + metrics."""
    try:
        if model_type == "bayesian_ridge":
            return _fit_bayesian_ridge(X_train, y_train, X_test, y_test)
        elif model_type == "gaussian_process":
            return _fit_gaussian_process(X_train, y_train, X_test, y_test)
        elif model_type == "quantile_regression":
            return _fit_quantile_regression(X_train, y_train, X_test, y_test)
        elif model_type == "gradient_boosting":
            return _fit_gradient_boosting(X_train, y_train, X_test, y_test)
        else:
            return {"error": f"Unknown model type '{model_type}'. Call GET /models for valid options."}
    except Exception as e:
        return {"error": f"Model fitting failed: {str(e)}"}


def predict_with_uncertainty(model_type, X_train, y_train, X_test, y_test):
    return fit_model(model_type, X_train, y_train, X_test, y_test)
