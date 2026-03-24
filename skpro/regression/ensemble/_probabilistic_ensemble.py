"""Probabilistic boosting and stacking compositors for skpro.

Implements ensemble compositors for boosting and stacking of probabilistic regressors.
"""

import numpy as np

from skpro.distributions.mixture import Mixture
from skpro.regression.base import BaseProbaRegressor


class ProbabilisticStackingRegressor(BaseProbaRegressor):
    """Stacking ensemble for probabilistic regressors.

    Fits multiple base probabilistic regressors and combines their predictions
    using a weighted mixture (default: uniform weights).

    This class is inspired by modular, composable stacking/meta-learning frameworks
    that produce calibrated probabilistic or quantile predictions, as described in:

    - Dudek, G. (2024). Stacking for Probabilistic Short-term Load Forecasting.
      https://arxiv.org/abs/2406.10718
    - Large, J., et al. (2019). A probabilistic classifier ensemble weighting scheme
      based on cross-validated accuracy estimates (CAWPE).
      Machine Learning. https://pmc.ncbi.nlm.nih.gov/articles/PMC6790343/

    Parameters
    ----------
    estimators : list of (str, BaseProbaRegressor)
        List of (name, estimator) tuples.
    weights : list of float, optional (default=None)
        Mixture weights for the base estimators. If None, uniform weights are used.
    """

    def __init__(
        self, estimators, weights=None, meta_learner=None, use_meta_proba=True
    ):
        """
        Initialize ProbabilisticStackingRegressor.

        Parameters
        ----------
        estimators : list of (str, BaseProbaRegressor)
            List of (name, estimator) tuples.
        weights : list of float, optional (default=None)
            Mixture weights for the base estimators. If None, uniform weights are used.
        meta_learner : regressor or classifier, optional
            If provided, fits a meta-learner on the base predictions (mean or proba).
            Should support fit(X, y) and predict or predict_proba.
        use_meta_proba : bool, default True
            If True and meta_learner supports predict_proba, for probabilistic output.
        """
        self.estimators = estimators
        self.weights = weights
        self.meta_learner = meta_learner
        self.use_meta_proba = use_meta_proba
        super().__init__()

    def set_meta_learner(self, meta_learner, use_meta_proba=True):
        """Set or update the meta-learner for stacking.

        Parameters
        ----------
        meta_learner : regressor or classifier
            The meta-learner to use for stacking.
        use_meta_proba : bool, default True
            Whether to use predict_proba if available.
        """
        self.meta_learner = meta_learner
        self.use_meta_proba = use_meta_proba
        return self

    def add_base_estimator(self, name, estimator):
        """Add a new base estimator to the ensemble.

        Parameters
        ----------
        name : str
            Name for the estimator.
        estimator : BaseProbaRegressor
            The estimator to add.
        """
        if not hasattr(self, "estimators") or self.estimators is None:
            self.estimators = []
        self.estimators.append((name, estimator))
        return self

    def _fit(self, X, y, C=None):
        self.fitted_estimators_ = []
        base_preds = []
        for name, est in self.estimators:
            est_fitted = est.clone().fit(X, y, C)
            self.fitted_estimators_.append((name, est_fitted))
            base_preds.append(est_fitted.predict(X))
        # If meta-learner is provided, fit it on base predictions
        if self.meta_learner is not None:
            # Stack base predictions as features
            X_meta = np.column_stack([p.values.flatten() for p in base_preds])
            y_meta = y.values.flatten() if hasattr(y, "values") else y
            self.meta_learner_ = self.meta_learner.fit(X_meta, y_meta)
        else:
            self.meta_learner_ = None
        return self

    def _predict_proba(self, X):
        dists = [(name, est.predict_proba(X)) for name, est in self.fitted_estimators_]
        weights = self.weights
        if weights is not None:
            weights = np.array(weights) / np.sum(weights)
        # If meta-learner is provided, use its output as the final prediction
        if self.meta_learner_ is not None:
            base_preds = [
                est.predict(X).values.flatten() for _, est in self.fitted_estimators_
            ]
            X_meta = np.column_stack(base_preds)
            if self.use_meta_proba and hasattr(self.meta_learner_, "predict_proba"):
                # Use meta-learner's predict_proba if available
                meta_pred = self.meta_learner_.predict_proba(X_meta)
                # Return as a Mixture with a single component for compatibility
                return Mixture(distributions=[("meta", meta_pred)], weights=[1.0])
            else:
                # Use meta-learner's predict (point prediction)
                meta_pred = self.meta_learner_.predict(X_meta)
                # Return as a degenerate Mixture
                return Mixture(distributions=[("meta", meta_pred)], weights=[1.0])
        # Always return a Mixture, even if meta_learner_ is None
        return Mixture(distributions=dists, weights=weights)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for ProbabilisticStackingRegressor."""
        from sklearn.linear_model import LinearRegression

        from skpro.regression.residual import ResidualDouble

        est1 = ResidualDouble(LinearRegression())
        est2 = ResidualDouble(LinearRegression())
        # Two parameter sets: one with two estimators, one with three and weights
        params1 = {"estimators": [("est1", est1), ("est2", est2)]}
        params2 = {
            "estimators": [("est1", est1), ("est2", est2), ("est3", est1)],
            "weights": [0.2, 0.3, 0.5],
        }
        return [params1, params2]


class ProbabilisticBoostingRegressor(BaseProbaRegressor):
    """Residual-based probabilistic boosting ensemble for skpro.

    Trains base probabilistic regressors sequentially, each on the residuals
    of the previous prediction (y_true - y_pred_mean).
    Final prediction is a weighted mixture of all base predictions,
    as a true boosting ensemble for probabilistic regression.

    Class inspired by modular, composable probabilistic boosting frameworks, including:

    - Mendonça, A., et al. (2022). ProBoost: a Boosting Method
    for Probabilistic Classifiers. https://arxiv.org/abs/2209.01611
    - Sprangers, P., Schelter, S., & de Rijke, M. (2021). Probabilistic Gradient
    Boosting Machines (PGBM) for Large-Scale Probabilistic Regression.
    https://arxiv.org/abs/2106.01682
    - Tu, Z. (2005). Probabilistic Boosting-Tree: Learning Discriminative Models
    for Classification, Recognition, and Clustering. ICCV 2005.
    https://pages.ucsd.edu/~ztu/publication/iccv05_pbt.pdf

    Parameters
    ----------
    base_estimator : BaseProbaRegressor
        The base probabilistic regressor to boost.
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float, optional (default=1.0)
        Shrinks the contribution of each regressor.
    """

    def __init__(
        self,
        base_estimator,
        n_estimators=10,
        learning_rate=1.0,
        uncertainty_weighting=None,
        calibrator=None,
    ):
        """
        Initialize ProbabilisticBoostingRegressor.

        Parameters
        ----------
        base_estimator : BaseProbaRegressor
            The base probabilistic regressor to boost.
        n_estimators : int
            Number of boosting rounds.
        learning_rate : float, optional (default=1.0)
            Shrinks the contribution of each regressor.
        uncertainty_weighting : callable or None
            If provided, a function that takes (y_true, y_pred, round_idx)
            and returns a weight for this round.
            Inspired by ProBoost and PGBM for uncertainty-aware weighting.
        calibrator : object or None
            If provided, should have fit(y_true, y_pred) and predict_proba(y_pred)
            for post-hoc calibration.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.uncertainty_weighting = uncertainty_weighting
        self.calibrator = calibrator
        super().__init__()

    def _fit(self, X, y, C=None):
        self.estimators_ = []
        self.weights_ = []
        residual = y.copy()
        for i in range(self.n_estimators):
            est = self.base_estimator.clone().fit(X, residual, C)
            self.estimators_.append(est)
            # Uncertainty-aware weighting (if provided)
            y_pred = est.predict(X)
            if self.uncertainty_weighting is not None:
                weight = self.uncertainty_weighting(y, y_pred, i)
            else:
                weight = self.learning_rate
            self.weights_.append(weight)
            # Update residuals: y_true - y_pred_mean
            if hasattr(y_pred, "to_frame"):
                y_pred = (
                    y_pred
                    if isinstance(y_pred, type(y))
                    else y_pred.to_frame(
                        index=y.index,
                        name=y.columns[0] if hasattr(y, "columns") else None,
                    )
                )
            residual = y - y_pred
        # Optionally fit a calibrator on the final ensemble output
        if self.calibrator is not None:
            # Fit calibrator on the final ensemble prediction
            y_pred_ensemble = self._predict_proba(X)
            self.calibrator_ = self.calibrator.fit(y, y_pred_ensemble)
        else:
            self.calibrator_ = None
        return self

    def _predict_proba(self, X):
        dists = [
            (f"est{i}", est.predict_proba(X)) for i, est in enumerate(self.estimators_)
        ]
        weights = np.array(self.weights_) / np.sum(self.weights_)
        mixture = Mixture(distributions=dists, weights=weights)
        # Optionally calibrate the output
        if self.calibrator_ is not None:
            return self.calibrator_.predict_proba(mixture)
        return mixture

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for ProbabilisticBoostingRegressor."""
        from sklearn.linear_model import LinearRegression

        from skpro.regression.residual import ResidualDouble

        base1 = ResidualDouble(LinearRegression())
        base2 = ResidualDouble(LinearRegression())
        # Two parameter sets: one with 3 estimators, one with 5 and different base
        params1 = {"base_estimator": base1, "n_estimators": 3}
        params2 = {"base_estimator": base2, "n_estimators": 5}
        return [params1, params2]
