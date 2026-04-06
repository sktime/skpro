"""Probabilistic boosting and stacking compositors for skpro."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from skpro.distributions.mixture import Mixture
from skpro.distributions.normal import Normal
from skpro.regression.base import BaseProbaRegressor
from skpro.regression.residual import ResidualDouble


class ProbabilisticStackingRegressor(BaseProbaRegressor):
    """Stacking ensemble for probabilistic regressors.

    - No meta_learner: weighted Mixture of base distributions.
    - With meta_learner: meta trained on stacked mean + variance features;
      returns meta's predict_proba (must return a proper skpro distribution).
    """

    _tags = {
        "authors": ["arnavk23"],
        "estimator_type": "regressor_proba",
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        params.update(
            {
                "estimators": self.estimators,
                "weights": self.weights,
                "meta_learner": self.meta_learner,
            }
        )
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    _tags = {
        "authors": ["arnavk23"],
        "estimator_type": "regressor_proba",
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, estimators=None, weights=None, meta_learner=None):
        if estimators is None:
            dummy = ResidualDouble(LinearRegression())
            estimators = [("dummy", dummy)]
        if not isinstance(estimators, list) or not all(
            isinstance(e, tuple) and len(e) == 2 for e in estimators
        ):
            raise ValueError(
                "estimators must be a list of (str, BaseProbaRegressor) tuples."
            )

        for _, est in estimators:
            if not hasattr(est, "predict_proba"):
                raise ValueError("Each base estimator must implement predict_proba.")

        if weights is not None:
            if len(weights) != len(estimators):
                raise ValueError("weights must have the same length as estimators.")
            if any(w < 0 for w in weights):
                raise ValueError("weights must be non-negative.")

        super().__init__()
        self.estimators = estimators
        self.weights = weights
        self.meta_learner = meta_learner

    def add_base_estimator(self, name, estimator):
        """Add base estimator; returns a new instance (no mutation of original)."""
        new_self = self.clone()
        new_self.estimators = (
            list(new_self.estimators) if new_self.estimators is not None else []
        )
        new_self.estimators.append((name, estimator))
        return new_self

    def _get_meta_features(self, X):
        """Stack mean + variance from each base estimator."""
        features = []
        for _, est in self.fitted_estimators_:
            dist = est.predict_proba(X)
            mean_vals = np.asarray(dist.mean()).flatten()

            if hasattr(dist, "var") and callable(dist.var):
                var_vals = np.asarray(dist.var()).flatten()
            else:
                var_vals = np.zeros_like(mean_vals)

            features.extend([mean_vals, var_vals])

        return np.column_stack(features)

    def _fit(self, X, y, C=None):
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            est_fitted = (
                est.clone().fit(X, y, C) if hasattr(est, "clone") else est.fit(X, y, C)
            )
            self.fitted_estimators_.append((name, est_fitted))

        # Save y columns for output contract
        if hasattr(y, "columns"):
            self._y_columns_ = list(y.columns)
        else:
            self._y_columns_ = ["target"]

        if self.meta_learner is not None:
            X_meta = self._get_meta_features(X)
            y_meta = np.asarray(y).flatten()
            meta = (
                self.meta_learner.clone()
                if hasattr(self.meta_learner, "clone")
                else self.meta_learner
            )
            self.meta_learner_ = meta.fit(X_meta, y_meta)
        else:
            self.meta_learner_ = None

        return self

    def _predict_proba(self, X):
        dist = None
        if hasattr(self, "meta_learner_") and self.meta_learner_ is not None:
            X_meta = self._get_meta_features(X)
            dist = self.meta_learner_.predict_proba(X_meta)
        else:
            # Mixture path
            dists = [
                (name, est.predict_proba(X)) for name, est in self.fitted_estimators_
            ]
            weights = None
            if self.weights is not None:
                weights = np.asarray(self.weights, dtype=float)
                weights = weights / weights.sum()
            dist = Mixture(distributions=dists, weights=weights)

        if hasattr(X, "index") and hasattr(dist, "index"):
            dist.index = X.index

        if hasattr(dist, "columns"):
            dist.columns = pd.Index(self._y_columns_)

        return dist

    def _predict(self, X):
        mean = self._predict_proba(X).mean()
        mean = np.asarray(mean)
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
        # Only use as many columns as y
        n_cols = len(self._y_columns_)
        if mean.shape[1] > n_cols:
            mean = mean[:, :n_cols]
        return pd.DataFrame(
            mean, columns=self._y_columns_, index=getattr(X, "index", None)
        )

    def _predict_interval(self, X, coverage=0.9):
        """Predict prediction intervals with correct shape and index."""
        dist = self._predict_proba(X)
        coverage_arr = np.atleast_1d(coverage)
        lower, upper = None, None
        can_vectorize_interval = not (
            isinstance(dist, Normal) and len(coverage_arr) > 1
        )
        if hasattr(dist, "interval") and can_vectorize_interval:
            lower, upper = dist.interval(coverage_arr)

        if lower is None or upper is None:
            mean = np.asarray(dist.mean())
            if isinstance(mean, pd.DataFrame):
                mean = mean.to_numpy()
            if mean.ndim == 1:
                mean = mean[:, np.newaxis]
            n_samples, n_vars = mean.shape
            n_cols = len(self._y_columns_)
            if n_vars > n_cols:
                mean = mean[:, :n_cols]
            sigma = np.full_like(mean, 1e-5, dtype=float)
            n_cov = len(coverage_arr)
            idx = getattr(X, "index", None)
            normal_dist = Normal(mu=mean, sigma=sigma, index=idx)
            lowers = []
            uppers = []
            for c in coverage_arr:
                lower_bound, upper_bound = normal_dist.interval(c)
                lower_bound = np.asarray(lower_bound)
                upper_bound = np.asarray(upper_bound)
                if lower_bound.ndim == 1:
                    lower_bound = lower_bound[:, np.newaxis]
                if upper_bound.ndim == 1:
                    upper_bound = upper_bound[:, np.newaxis]
                lowers.append(lower_bound)
                uppers.append(upper_bound)
            lower = np.stack(lowers, axis=-1)  # (n_samples, n_cols, n_cov)
            upper = np.stack(uppers, axis=-1)

        # Ensure lower/upper are arrays with shape
        # (n_samples, n_vars, n_cov)
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        # Expand dims if needed
        if lower.ndim == 1:
            lower = lower[:, np.newaxis, np.newaxis]
            upper = upper[:, np.newaxis, np.newaxis]
        elif lower.ndim == 2:
            lower = lower[:, :, np.newaxis]
            upper = upper[:, :, np.newaxis]
        n_samples, n_vars, n_cov = lower.shape
        # Stack lower/upper along last axis (bound)
        data = np.stack([lower, upper], axis=3)  # (n_samples, n_vars, n_cov, 2)
        # Reshape to (n_samples, n_vars * n_cov * 2)
        data = data.reshape(n_samples, n_vars * n_cov * 2)
        bounds = ["lower", "upper"]
        var_names = (
            self._y_columns_
            if hasattr(self, "_y_columns_")
            else ([0] if n_vars == 1 else list(range(n_vars)))
        )
        # Always build 3-level MultiIndex: (variable, coverage, bound)
        columns = pd.MultiIndex.from_tuples(
            [(v, c, b) for v in var_names for c in coverage_arr for b in bounds],
            names=["variable", "coverage", "bound"],
        )
        return pd.DataFrame(data, columns=columns, index=getattr(X, "index", None))

    def _predict_quantiles(self, X, alpha):
        dist = self._predict_proba(X)
        if hasattr(dist, "quantile"):
            return dist.quantile(alpha)

        # Fallback
        mean = np.asarray(dist.mean())
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
        sigma = np.full_like(mean, 1e-5, dtype=float)
        idx = getattr(X, "index", None)
        cols = getattr(self, "_y_columns_", None)
        return Normal(mu=mean, sigma=sigma, index=idx, columns=cols).quantile(alpha)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for automated tests."""
        from sklearn.linear_model import LinearRegression

        from skpro.regression.residual import ResidualDouble

        base = ResidualDouble(LinearRegression())
        meta = ResidualDouble(LinearRegression())

        return [
            {"estimators": [("est1", base), ("est2", base)]},
            {
                "estimators": [("est1", base), ("est2", base)],
                "weights": [0.4, 0.6],
                "meta_learner": meta,
            },
        ]


class ProbabilisticBoostingRegressor(BaseProbaRegressor):
    """Residual-based probabilistic boosting ensemble.

    Fits bases sequentially on residuals of previous mean prediction.
    Final output: weighted mixture of all base distributions.
    """

    _tags = {
        "authors": ["arnavk23"],
        "estimator_type": "regressor_proba",
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        learning_rate=1.0,
        uncertainty_weighting=None,
        calibrator=None,
    ):
        if base_estimator is None:
            base_estimator = ResidualDouble(LinearRegression())
        if not hasattr(base_estimator, "predict_proba"):
            raise ValueError("base_estimator must implement predict_proba.")
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        super().__init__()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.uncertainty_weighting = uncertainty_weighting
        self.calibrator = calibrator

        super().__init__()

    def _fit(self, X, y, C=None):
        self.estimators_ = []
        self.weights_ = []

        y_arr = np.asarray(y).flatten()
        residual = (
            y.copy()
            if hasattr(y, "copy")
            else pd.DataFrame({"residual": y_arr}, index=getattr(y, "index", None))
        )

        # Save y columns for output contract
        if hasattr(y, "columns"):
            self._y_columns_ = list(y.columns)
        else:
            self._y_columns_ = ["target"]

        for i in range(self.n_estimators):
            est = (
                self.base_estimator.clone().fit(X, residual, C)
                if hasattr(self.base_estimator, "clone")
                else self.base_estimator.fit(X, residual, C)
            )
            self.estimators_.append(est)

            y_pred = est.predict(X)
            y_pred_arr = np.asarray(y_pred).flatten()

            weight = (
                self.uncertainty_weighting(y, y_pred, i)
                if self.uncertainty_weighting is not None
                else self.learning_rate
            )
            self.weights_.append(weight)

            residual_arr = y_arr - y_pred_arr
            if hasattr(y, "index"):
                residual = pd.DataFrame(residual_arr, index=y.index)
            else:
                residual = residual_arr

        if self.calibrator is not None:
            final_dist = self._predict_proba(X)
            cal = (
                self.calibrator.clone()
                if hasattr(self.calibrator, "clone")
                else self.calibrator
            )
            self.calibrator_ = cal.fit(y, final_dist)
        else:
            self.calibrator_ = None

        return self

    def _predict_proba(self, X):
        dists = [
            (f"est{i}", est.predict_proba(X)) for i, est in enumerate(self.estimators_)
        ]
        weights = np.asarray(self.weights_, dtype=float)
        weights = weights / weights.sum()

        mixture = Mixture(distributions=dists, weights=weights)

        if hasattr(X, "index") and hasattr(mixture, "index"):
            mixture.index = X.index

        if hasattr(mixture, "columns"):
            mixture.columns = pd.Index(self._y_columns_)

        if hasattr(self, "calibrator_") and self.calibrator_ is not None:
            calibrated = self.calibrator_.predict_proba(mixture)
            if hasattr(X, "index") and hasattr(calibrated, "index"):
                calibrated.index = X.index
            if hasattr(calibrated, "columns"):
                calibrated.columns = pd.Index(self._y_columns_)
            return calibrated
        return mixture

    def _predict(self, X):
        mean = self._predict_proba(X).mean()
        mean = np.asarray(mean)
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
        n_cols = len(self._y_columns_)
        if mean.shape[1] > n_cols:
            mean = mean[:, :n_cols]
        return pd.DataFrame(
            mean, columns=self._y_columns_, index=getattr(X, "index", None)
        )

    def _predict_interval(self, X, coverage=0.9):
        """Predict prediction intervals with correct shape and index."""
        dist = self._predict_proba(X)
        coverage_arr = np.atleast_1d(coverage)
        lower, upper = None, None
        can_vectorize_interval = not (
            isinstance(dist, Normal) and len(coverage_arr) > 1
        )
        if hasattr(dist, "interval") and can_vectorize_interval:
            lower, upper = dist.interval(coverage_arr)

        if lower is None or upper is None:
            mean = np.asarray(dist.mean())
            if isinstance(mean, pd.DataFrame):
                mean = mean.to_numpy()
            if mean.ndim == 1:
                mean = mean[:, np.newaxis]
            n_samples, n_vars = mean.shape
            n_cols = len(self._y_columns_)
            if n_vars > n_cols:
                mean = mean[:, :n_cols]
            sigma = np.full_like(mean, 1e-5, dtype=float)
            n_cov = len(coverage_arr)
            idx = getattr(X, "index", None)
            normal_dist = Normal(mu=mean, sigma=sigma, index=idx)
            lowers = []
            uppers = []
            for c in coverage_arr:
                lower_bound, upper_bound = normal_dist.interval(c)
                lower_bound = np.asarray(lower_bound)
                upper_bound = np.asarray(upper_bound)
                if lower_bound.ndim == 1:
                    lower_bound = lower_bound[:, np.newaxis]
                if upper_bound.ndim == 1:
                    upper_bound = upper_bound[:, np.newaxis]
                lowers.append(lower_bound)
                uppers.append(upper_bound)
            lower = np.stack(lowers, axis=-1)  # (n_samples, n_cols, n_cov)
            upper = np.stack(uppers, axis=-1)

        lower = np.asarray(lower)
        upper = np.asarray(upper)
        if lower.ndim == 1:
            lower = lower[:, np.newaxis, np.newaxis]
            upper = upper[:, np.newaxis, np.newaxis]
        elif lower.ndim == 2:
            lower = lower[:, :, np.newaxis]
            upper = upper[:, :, np.newaxis]
        n_samples, n_vars, n_cov = lower.shape
        data = np.stack([lower, upper], axis=3)  # (n_samples, n_vars, n_cov, 2)
        data = data.reshape(n_samples, n_vars * n_cov * 2)
        bounds = ["lower", "upper"]
        var_names = (
            self._y_columns_
            if hasattr(self, "_y_columns_")
            else ([0] if n_vars == 1 else list(range(n_vars)))
        )
        columns = pd.MultiIndex.from_tuples(
            [(v, c, b) for v in var_names for c in coverage_arr for b in bounds],
            names=["variable", "coverage", "bound"],
        )
        return pd.DataFrame(data, columns=columns, index=getattr(X, "index", None))

    def _predict_quantiles(self, X, alpha):
        dist = self._predict_proba(X)
        if hasattr(dist, "quantile"):
            return dist.quantile(alpha)

        mean = np.asarray(dist.mean())
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
        sigma = np.full_like(mean, 1e-5, dtype=float)
        idx = getattr(X, "index", None)
        cols = getattr(self, "_y_columns_", None)
        return Normal(mu=mean, sigma=sigma, index=idx, columns=cols).quantile(alpha)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        params.update(
            {
                "base_estimator": self.base_estimator,
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "uncertainty_weighting": self.uncertainty_weighting,
                "calibrator": self.calibrator,
            }
        )
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for automated tests."""
        from sklearn.linear_model import LinearRegression

        from skpro.regression.residual import ResidualDouble

        base = ResidualDouble(LinearRegression())
        return [
            {"base_estimator": base, "n_estimators": 3},
            {"base_estimator": base, "n_estimators": 5, "learning_rate": 0.8},
        ]
