"""Tests for the Bayesian probabilistic regression interface."""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from skpro.distributions.base import BaseDistribution
from skpro.regression.base import BaseProbaRegressor
from skpro.regression.bayesian import (
    BaseBayesianRegressor,
    BayesianLinearClosedFormRegressor,
)

_PYMC_EXTRAS_AVAILABLE = _check_soft_dependencies("pymc_extras", severity="none")

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_data(n=30, n_features=2, seed=0, col_prefix="x", index_start=0):
    """Return (X, y) DataFrames with sequential RangeIndex starting at index_start."""
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(index_start, index_start + n)
    cols = [f"{col_prefix}{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.standard_normal((n, n_features)), columns=cols, index=idx)
    coefs = rng.standard_normal(n_features)
    noise = rng.standard_normal(n) * 0.1
    y_vals = X.to_numpy() @ coefs + noise
    y = pd.DataFrame({"target": y_vals}, index=idx)
    return X, y


def _make_linear_data(slope=2.0, intercept=1.0, n=50, noise_std=0.05, seed=42):
    """Return (X, y) with y ≈ intercept + slope * x (single feature)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y_vals = intercept + slope * x + rng.standard_normal(n) * noise_std
    X = pd.DataFrame({"x": x})
    y = pd.DataFrame({"y": y_vals})
    return X, y


# ---------------------------------------------------------------------------
# Stubs used for unit-testing hook dispatch WITHOUT PyMC
# ---------------------------------------------------------------------------


class _FakePredictions(dict):
    """Dict subclass that also exposes a ``data_vars`` property."""

    @property
    def data_vars(self):
        return self


class _FakeTrace:
    def __init__(self, predictions):
        self.predictions = predictions

    def groups(self):
        return ["predictions"]


class _MinimalNonMCRegressor(BaseBayesianRegressor):
    """Minimal subclass that overrides posterior hooks without PyMC/MCMC."""

    def _fit_posterior(self, X, y):
        """Store training mean as the only fitted attribute."""
        self._y_columns = y.columns.tolist()
        self._mu_train = y.mean().values  # shape (1,)

    def _predict_proba_from_posterior(self, X):
        """Always predict a Normal centred on training mean, unit variance."""
        from skpro.distributions import Normal

        mu = pd.DataFrame(
            np.full((len(X), 1), self._mu_train[0]),
            index=X.index,
            columns=self._y_columns,
        )
        sigma = pd.DataFrame(
            np.ones((len(X), 1)),
            index=X.index,
            columns=self._y_columns,
        )
        return Normal(mu=mu, sigma=sigma, index=X.index, columns=self._y_columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {}


# ===========================================================================
# 1. Module / import / class-hierarchy contract
# ===========================================================================


class TestImportsAndHierarchy:
    def test_all_public_names_importable(self):
        """Every name in __all__ should be importable without error."""
        import skpro.regression.bayesian as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"'{name}' listed in __all__ but not importable"

    def test_base_class_is_proba_regressor(self):
        assert issubclass(BaseBayesianRegressor, BaseProbaRegressor)

    def test_closed_form_is_bayesian_regressor(self):
        assert issubclass(BayesianLinearClosedFormRegressor, BaseBayesianRegressor)

    def test_object_type_tag(self):
        assert BaseBayesianRegressor.get_class_tag("object_type") == "regressor_proba"


# ===========================================================================
# 2. BaseBayesianRegressor constructor defaults and sample_kwargs
# ===========================================================================


class TestBaseBayesianRegressorConstructor:
    def test_default_parameter_values(self):
        reg = BaseBayesianRegressor()
        assert reg.draws == 1000
        assert reg.tune == 1000
        assert reg.chains == 2
        assert reg.target_accept == 0.95
        assert reg.random_seed is None
        assert reg.progressbar is True
        assert reg.sample_kwargs == {}

    def test_custom_parameters_stored(self):
        reg = BaseBayesianRegressor(
            draws=200,
            tune=50,
            chains=1,
            target_accept=0.8,
            random_seed=7,
            progressbar=False,
            sample_kwargs={"cores": 2},
        )
        assert reg.draws == 200
        assert reg.tune == 50
        assert reg.chains == 1
        assert reg.target_accept == 0.8
        assert reg.random_seed == 7
        assert reg.progressbar is False
        assert reg.sample_kwargs == {"cores": 2}

    def test_sample_kwargs_none_becomes_empty_dict(self):
        reg = BaseBayesianRegressor(sample_kwargs=None)
        assert reg.sample_kwargs == {}

    def test_build_model_raises_not_implemented(self):
        """Direct instantiation of base should raise when _build_model is called."""
        reg = BaseBayesianRegressor()
        X, y = _make_data(n=5)
        with pytest.raises(NotImplementedError):
            reg._build_model(X, y)


# ===========================================================================
# 3. _get_predictive_variable_name (all branches)
# ===========================================================================


class TestGetPredictiveVariableName:
    def _make_reg(self, predictions):
        reg = BaseBayesianRegressor()
        reg.trace_ = _FakeTrace(_FakePredictions(predictions))
        return reg

    def test_returns_y_obs_when_present(self):
        reg = self._make_reg({"y_obs": object(), "mu": object()})
        assert reg._get_predictive_variable_name() == "y_obs"

    def test_returns_single_var_when_no_y_obs(self):
        reg = self._make_reg({"pred_out": object()})
        assert reg._get_predictive_variable_name() == "pred_out"

    def test_configured_name_takes_precedence_over_y_obs(self):
        reg = self._make_reg({"y_obs": object(), "custom": object()})
        reg._predictive_var_name = "custom"
        assert reg._get_predictive_variable_name() == "custom"

    def test_configured_name_missing_raises(self):
        reg = self._make_reg({"y_obs": object()})
        reg._predictive_var_name = "no_such_var"
        with pytest.raises(ValueError, match="no_such_var"):
            reg._get_predictive_variable_name()

    def test_ambiguous_raises(self):
        reg = self._make_reg({"a": object(), "b": object()})
        with pytest.raises(ValueError, match="Could not infer predictive variable"):
            reg._get_predictive_variable_name()

    def test_no_trace_attr_raises(self):
        reg = BaseBayesianRegressor()  # trace_ not set
        with pytest.raises(ValueError):
            reg._get_predictive_variable_name()

    def test_predictions_group_missing_raises(self):
        class _TraceNoPredictions:
            def groups(self):
                return ["posterior"]

        reg = BaseBayesianRegressor()
        reg.trace_ = _TraceNoPredictions()
        with pytest.raises(ValueError, match="No 'predictions' group"):
            reg._get_predictive_variable_name()


# ===========================================================================
# 4. get_posterior_summary fallback
# ===========================================================================


class TestPosteriorSummaryFallback:
    def test_raises_not_implemented_without_trace(self):
        reg = BaseBayesianRegressor()
        # trace_ not set, should raise NotImplementedError with helpful message
        with pytest.raises(NotImplementedError, match="Override"):
            reg.get_posterior_summary()


# ===========================================================================
# 5. Hook contract: non-MC subclass end-to-end
# ===========================================================================


class TestNonMCHookContract:
    """Prove that the hook layer is correctly wired without any PyMC dependency."""

    def _fit_predict(self):
        X_tr, y_tr = _make_data(n=20)
        X_te, _ = _make_data(n=8, seed=99, index_start=20)
        reg = _MinimalNonMCRegressor()
        reg.fit(X_tr, y_tr)
        pred = reg.predict_proba(X_te)
        return reg, X_tr, y_tr, X_te, pred

    def test_is_fitted_after_fit(self):
        reg, X_tr, y_tr, *_ = self._fit_predict()
        assert reg.is_fitted

    def test_predict_proba_returns_distribution(self):
        *_, pred = self._fit_predict()
        assert isinstance(pred, BaseDistribution)

    def test_predict_proba_shape(self):
        _, X_tr, y_tr, X_te, pred = self._fit_predict()
        assert pred.shape == (len(X_te), 1)

    def test_predict_proba_index_matches_X_test(self):
        _, _, _, X_te, pred = self._fit_predict()
        assert list(pred.index) == list(X_te.index)

    def test_predict_proba_columns_match_y_columns(self):
        _, _, y_tr, _, pred = self._fit_predict()
        assert list(pred.columns) == list(y_tr.columns)

    def test_build_model_never_called(self):
        """_build_model would raise NotImplementedError; if it's never hit, we pass."""
        # _MinimalNonMCRegressor does NOT override _build_model, so any call
        # to it would propagate to the base which raises NotImplementedError.
        # A successful predict confirms _build_model was bypassed.
        *_, pred = self._fit_predict()
        assert pred is not None


# ===========================================================================
# 6. BayesianLinearClosedFormRegressor: fit / predict cycle
# ===========================================================================


class TestClosedFormFitPredict:
    """Full input/output contract tests for the closed-form estimator."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        X, y = _make_data(n=40, n_features=3, seed=1)
        X_te, _ = _make_data(n=10, n_features=3, seed=2, index_start=40)
        self.X, self.y, self.X_te = X, y, X_te
        self.reg = BayesianLinearClosedFormRegressor()
        self.reg.fit(X, y)

    def test_is_fitted(self):
        assert self.reg.is_fitted

    # --- predict_proba ---
    def test_predict_proba_returns_base_distribution(self):
        pred = self.reg.predict_proba(self.X_te)
        assert isinstance(pred, BaseDistribution)

    def test_predict_proba_shape(self):
        pred = self.reg.predict_proba(self.X_te)
        assert pred.shape == (len(self.X_te), 1)

    def test_predict_proba_index_matches_X_test(self):
        pred = self.reg.predict_proba(self.X_te)
        assert list(pred.index) == list(self.X_te.index)

    def test_predict_proba_columns_match_y_columns(self):
        pred = self.reg.predict_proba(self.X_te)
        assert list(pred.columns) == list(self.y.columns)

    # --- predict (mean) ---
    def test_predict_returns_dataframe(self):
        y_pred = self.reg.predict(self.X_te)
        assert isinstance(y_pred, pd.DataFrame)

    def test_predict_shape(self):
        y_pred = self.reg.predict(self.X_te)
        assert y_pred.shape == (len(self.X_te), 1)

    def test_predict_index_matches_X_test(self):
        y_pred = self.reg.predict(self.X_te)
        assert list(y_pred.index) == list(self.X_te.index)

    def test_predict_columns_match_y_columns(self):
        y_pred = self.reg.predict(self.X_te)
        assert list(y_pred.columns) == list(self.y.columns)

    def test_predict_equals_predict_proba_mean(self):
        y_pred = self.reg.predict(self.X_te)
        y_pred_proba_mean = self.reg.predict_proba(self.X_te).mean()
        pd.testing.assert_frame_equal(y_pred, y_pred_proba_mean, rtol=1e-10)

    # --- predict_interval ---
    def test_predict_interval_shape(self):
        from skpro.datatypes import check_raise

        pi = self.reg.predict_interval(self.X_te)
        check_raise(pi, "pred_interval", "Proba", "predict_interval return")

    def test_predict_interval_index_matches_X_test(self):
        pi = self.reg.predict_interval(self.X_te)
        assert list(pi.index) == list(self.X_te.index)

    def test_predict_interval_lower_le_upper(self):
        pi = self.reg.predict_interval(self.X_te)
        col = list(self.y.columns)[0]
        lower = pi[(col, 0.9, "lower")]
        upper = pi[(col, 0.9, "upper")]
        assert (lower <= upper).all()

    # --- predict_quantiles ---
    def test_predict_quantiles_shape(self):
        from skpro.datatypes import check_raise

        pq = self.reg.predict_quantiles(self.X_te, alpha=[0.1, 0.5, 0.9])
        check_raise(pq, "pred_quantiles", "Proba", "predict_quantiles return")

    def test_predict_quantiles_monotone(self):
        pq = self.reg.predict_quantiles(self.X_te, alpha=[0.1, 0.5, 0.9])
        col = list(self.y.columns)[0]
        q10 = pq[(col, 0.1)].values
        q50 = pq[(col, 0.5)].values
        q90 = pq[(col, 0.9)].values
        assert (q10 <= q50).all()
        assert (q50 <= q90).all()

    # --- predict_var ---
    def test_predict_var_positive(self):
        pv = self.reg.predict_var(self.X_te)
        assert isinstance(pv, pd.DataFrame)
        assert (pv.values > 0).all()

    def test_predict_var_index_matches(self):
        pv = self.reg.predict_var(self.X_te)
        assert list(pv.index) == list(self.X_te.index)


# ===========================================================================
# 7. BayesianLinearClosedFormRegressor: non-default index
# ===========================================================================


class TestClosedFormNonDefaultIndex:
    def test_non_sequential_integer_index_preserved(self):
        """predict_proba.index must equal X_test.index when using a funny index."""
        idx_tr = pd.Index([10, 20, 30, 40, 50])
        idx_te = pd.Index([100, 200, 300])
        X_tr = pd.DataFrame({"a": np.arange(5, dtype=float)}, index=idx_tr)
        y_tr = pd.DataFrame({"y": np.arange(5, dtype=float) * 2}, index=idx_tr)
        X_te = pd.DataFrame({"a": [6.0, 7.0, 8.0]}, index=idx_te)

        reg = BayesianLinearClosedFormRegressor()
        reg.fit(X_tr, y_tr)
        pred = reg.predict_proba(X_te)

        assert list(pred.index) == list(idx_te)

    def test_string_column_name_preserved(self):
        X, y = _make_data(n=10, n_features=1)
        y = y.rename(columns={"target": "my_custom_target"})
        reg = BayesianLinearClosedFormRegressor()
        reg.fit(X, y)
        pred = reg.predict_proba(X)
        assert list(pred.columns) == ["my_custom_target"]


# ===========================================================================
# 8. BayesianLinearClosedFormRegressor: fit_intercept flag
# ===========================================================================


class TestClosedFormFitIntercept:
    @pytest.fixture
    def data(self):
        return _make_data(n=20, n_features=2)

    def test_with_intercept_coef_names_include_intercept(self, data):
        X, y = data
        reg = BayesianLinearClosedFormRegressor(fit_intercept=True)
        reg.fit(X, y)
        assert "intercept" in reg.coef_names_

    def test_with_intercept_n_coefs(self, data):
        X, y = data
        reg = BayesianLinearClosedFormRegressor(fit_intercept=True)
        reg.fit(X, y)
        # n_features + 1 intercept
        assert len(reg.coef_names_) == X.shape[1] + 1

    def test_without_intercept_coef_names_no_intercept(self, data):
        X, y = data
        reg = BayesianLinearClosedFormRegressor(fit_intercept=False)
        reg.fit(X, y)
        assert "intercept" not in reg.coef_names_

    def test_without_intercept_n_coefs(self, data):
        X, y = data
        reg = BayesianLinearClosedFormRegressor(fit_intercept=False)
        reg.fit(X, y)
        assert len(reg.coef_names_) == X.shape[1]

    def test_fit_intercept_false_predict_proba_correct_shape(self, data):
        X, y = data
        reg = BayesianLinearClosedFormRegressor(fit_intercept=False)
        reg.fit(X, y)
        pred = reg.predict_proba(X)
        assert pred.shape == y.shape


# ===========================================================================
# 9. Posterior structural properties
# ===========================================================================


class TestPosteriorStructure:
    @pytest.fixture(autouse=True)
    def _setup(self):
        X, y = _make_data(n=30, n_features=2, seed=5)
        self.reg = BayesianLinearClosedFormRegressor(fit_intercept=True)
        self.reg.fit(X, y)
        self.n_coef = X.shape[1] + 1  # 2 features + intercept

    def test_posterior_mu_shape(self):
        assert self.reg.posterior_mu_.shape == (self.n_coef, 1)

    def test_posterior_cov_shape(self):
        assert self.reg.posterior_cov_.shape == (self.n_coef, self.n_coef)

    def test_posterior_cov_symmetric(self):
        cov = self.reg.posterior_cov_
        assert np.allclose(cov, cov.T, atol=1e-12)

    def test_posterior_cov_positive_definite(self):
        eigs = np.linalg.eigvalsh(self.reg.posterior_cov_)
        assert (
            eigs > 0
        ).all(), f"Posterior covariance not PD; min eigenvalue={eigs.min()}"

    def test_coef_names_length(self):
        assert len(self.reg.coef_names_) == self.n_coef

    def test_coef_names_first_is_intercept(self):
        assert self.reg.coef_names_[0] == "intercept"


# ===========================================================================
# 10. Posterior accessors
# ===========================================================================


class TestPosteriorAccessors:
    @pytest.fixture(autouse=True)
    def _setup(self):
        X, y = _make_data(n=20, n_features=2, seed=7)
        self.reg = BayesianLinearClosedFormRegressor()
        self.reg.fit(X, y)

    def test_get_fitted_params_returns_dict(self):
        params = self.reg.get_fitted_params()
        assert isinstance(params, dict)

    def test_get_fitted_params_expected_keys(self):
        params = self.reg.get_fitted_params()
        for key in ("posterior_mu_", "posterior_cov_", "coef_names_"):
            assert (
                key in params
            ), f"Expected key '{key}' missing from get_fitted_params()"

    def test_get_posterior_summary_returns_dataframe(self):
        summary = self.reg.get_posterior_summary()
        assert isinstance(summary, pd.DataFrame)

    def test_get_posterior_summary_columns(self):
        summary = self.reg.get_posterior_summary()
        assert list(summary.columns) == ["mean", "std"]

    def test_get_posterior_summary_index_matches_coef_names(self):
        summary = self.reg.get_posterior_summary()
        assert list(summary.index) == list(self.reg.coef_names_)

    def test_get_posterior_summary_std_positive(self):
        summary = self.reg.get_posterior_summary()
        assert (summary["std"] > 0).all()


# ===========================================================================
# 11. Analytical correctness
# ===========================================================================


class TestAnalyticalCorrectness:
    """Verify that the closed-form update matches known conjugate results."""

    def test_recovers_slope_for_simple_linear_data(self):
        """With tight noise, posterior mean slope should be close to true slope."""
        slope, intercept = 2.5, -0.5
        X, y = _make_linear_data(
            slope=slope, intercept=intercept, n=200, noise_std=0.02, seed=1
        )
        reg = BayesianLinearClosedFormRegressor(
            prior_mean=0.0,
            prior_precision=0.001,  # very weak prior
            noise_precision=2500.0,  # matches noise_std=0.02  (1/0.02^2)
            fit_intercept=True,
        )
        reg.fit(X, y)
        est_intercept = reg.posterior_mu_[0, 0]
        est_slope = reg.posterior_mu_[1, 0]
        assert abs(est_slope - slope) < 0.1, f"Slope {est_slope:.4f} != {slope}"
        assert (
            abs(est_intercept - intercept) < 0.1
        ), f"Intercept {est_intercept:.4f} != {intercept}"

    def test_no_intercept_slope_recovery(self):
        """With fit_intercept=False, posterior mu[0] ≈ true slope."""
        rng = np.random.default_rng(10)
        x = rng.uniform(0, 2, 150)
        y_vals = 3.0 * x + rng.standard_normal(150) * 0.05
        X = pd.DataFrame({"x": x})
        y = pd.DataFrame({"y": y_vals})
        reg = BayesianLinearClosedFormRegressor(
            prior_mean=0.0,
            prior_precision=0.001,
            noise_precision=400.0,
            fit_intercept=False,
        )
        reg.fit(X, y)
        assert abs(reg.posterior_mu_[0, 0] - 3.0) < 0.1

    def test_prior_dominates_with_zero_data(self):
        """With 1 data point and very strong prior, posterior ≈ prior mean."""
        X = pd.DataFrame({"x": [100.0]})
        y = pd.DataFrame({"y": [100.0]})
        prior_mean = 5.0
        reg = BayesianLinearClosedFormRegressor(
            prior_mean=prior_mean,
            prior_precision=1e6,  # extremely strong prior
            noise_precision=0.001,  # very noisy observation
            fit_intercept=False,
        )
        reg.fit(X, y)
        assert abs(reg.posterior_mu_[0, 0] - prior_mean) < 0.5

    def test_data_dominates_with_weak_prior(self):
        """With many data points and weak prior, posterior mean ≈ MLE."""
        X, y = _make_linear_data(
            slope=4.0, intercept=0.0, n=500, noise_std=0.01, seed=13
        )
        reg = BayesianLinearClosedFormRegressor(
            prior_mean=0.0,
            prior_precision=1e-6,
            noise_precision=10000.0,
            fit_intercept=False,
        )
        reg.fit(X, y)
        assert abs(reg.posterior_mu_[0, 0] - 4.0) < 0.05

    def test_posterior_mean_prediction_close_to_insample_y(self):
        """E[y|x] from predict should approximate y_train closely for low noise."""
        X, y = _make_linear_data(
            slope=1.0, intercept=0.0, n=100, noise_std=0.01, seed=20
        )
        reg = BayesianLinearClosedFormRegressor(
            prior_precision=0.001,
            noise_precision=10000.0,
            fit_intercept=True,
        )
        reg.fit(X, y)
        y_hat = reg.predict(X)
        residuals = (y.values - y_hat.values).ravel()
        assert np.abs(residuals).mean() < 0.05

    def test_predictive_variance_everywhere_positive(self):
        X, y = _make_data(n=20)
        reg = BayesianLinearClosedFormRegressor()
        reg.fit(X, y)
        pv = reg.predict_var(X)
        assert (pv.values > 0).all()

    def test_predictive_variance_decreases_with_more_data(self):
        """More training data → tighter posterior → lower predictive variance."""
        X_te = pd.DataFrame({"x": [0.5, 1.0, 1.5]})

        def _fit_predict_var(n):
            rng = np.random.default_rng(0)
            x = rng.uniform(0, 2, n)
            y_vals = 2.0 * x + rng.standard_normal(n) * 0.3
            X_tr = pd.DataFrame({"x": x})
            y_tr = pd.DataFrame({"y": y_vals})
            reg = BayesianLinearClosedFormRegressor(
                prior_precision=1.0, noise_precision=11.0, fit_intercept=False
            )
            reg.fit(X_tr, y_tr)
            return reg.predict_var(X_te).values.mean()

        var_small = _fit_predict_var(5)
        var_large = _fit_predict_var(200)
        assert var_large < var_small, (
            f"Predictive variance should shrink with more data: "
            f"n=5 → {var_small:.4f}, n=200 → {var_large:.4f}"
        )


# ===========================================================================
# 12. Generalisation: predict on held-out data
# ===========================================================================


class TestGeneralisation:
    def test_predict_on_held_out_x_shape(self):
        X_tr, y_tr = _make_data(n=30, n_features=2, seed=1)
        X_te, _ = _make_data(n=12, n_features=2, seed=99, index_start=30)
        reg = BayesianLinearClosedFormRegressor()
        reg.fit(X_tr, y_tr)
        pred = reg.predict_proba(X_te)
        assert pred.shape == (12, 1)
        assert list(pred.index) == list(X_te.index)

    def test_repeated_predict_proba_gives_same_result(self):
        """predict_proba is deterministic for the closed-form estimator."""
        X, y = _make_data(n=20, seed=3)
        reg = BayesianLinearClosedFormRegressor()
        reg.fit(X, y)
        p1 = reg.predict_proba(X).mean()
        p2 = reg.predict_proba(X).mean()
        pd.testing.assert_frame_equal(p1, p2)

    def test_predict_proba_different_on_ood_x(self):
        """Predictions on out-of-distribution X should differ from training mean."""
        X_tr = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        y_tr = pd.DataFrame({"y": [0.0, 2.0, 4.0]})
        reg = BayesianLinearClosedFormRegressor(
            prior_precision=0.01, noise_precision=1000.0, fit_intercept=False
        )
        reg.fit(X_tr, y_tr)
        # Far extrapolation: expect predicted mean ≈ 2 * 100 = 200, not training mean
        X_far = pd.DataFrame({"x": [100.0]})
        pred_mean = reg.predict(X_far).iloc[0, 0]
        train_mean = y_tr["y"].mean()
        assert abs(pred_mean - train_mean) > 50


# ===========================================================================
# 13. get_test_params validity
# ===========================================================================


class TestGetTestParams:
    def test_both_param_sets_create_valid_instances(self):
        for params in BayesianLinearClosedFormRegressor.get_test_params():
            reg = BayesianLinearClosedFormRegressor(**params)
            X, y = _make_data(n=10, n_features=2)
            reg.fit(X, y)
            pred = reg.predict_proba(X)
            assert pred.shape == y.shape


# ===========================================================================
# 14. BayesianLinearRegressor constructor (structural, no fitting)
#     Skipped when pymc-extras is not installed
# ===========================================================================


@pytest.mark.skipif(
    not _PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
class TestBayesianLinearRegressorConstructor:
    from skpro.regression.bayesian import BayesianLinearRegressor

    def _make_reg(self, sampler_config=None, prior_config=None):
        from skpro.regression.bayesian import BayesianLinearRegressor

        return BayesianLinearRegressor(
            sampler_config=sampler_config or {},
            prior_config=prior_config or {},
        )

    def test_standard_sampler_keys_map_to_base_params(self):
        reg = self._make_reg(sampler_config={"draws": 50, "chains": 1})
        assert reg.draws == 50
        assert reg.chains == 1

    def test_extra_sampler_keys_forwarded_to_sample_kwargs(self):
        """Unknown sampler keys (e.g. 'cores') must land in sample_kwargs."""
        reg = self._make_reg(sampler_config={"cores": 3, "nuts_sampler": "numpyro"})
        assert reg.sample_kwargs.get("cores") == 3
        assert reg.sample_kwargs.get("nuts_sampler") == "numpyro"

    def test_extra_sampler_keys_not_in_base_params(self):
        reg = self._make_reg(sampler_config={"cores": 3})
        # 'cores' should NOT appear as a top-level attribute
        assert not hasattr(reg, "cores")

    def test_default_sampler_params_complete(self):
        """default_sampler_config must contain the six standard keys."""
        from skpro.regression.bayesian import BayesianLinearRegressor

        dsc = BayesianLinearRegressor.default_sampler_config.fget(
            BayesianLinearRegressor
        )
        for key in (
            "draws",
            "tune",
            "chains",
            "target_accept",
            "random_seed",
            "progressbar",
        ):
            assert key in dsc, f"'{key}' missing from default_sampler_config"

    def test_prior_config_override_intercept(self):
        from pymc_extras.prior import Prior

        reg = self._make_reg(prior_config={"intercept": Prior("Normal", mu=5, sigma=1)})
        # Custom prior should survive into self.prior_config
        assert reg.prior_config["intercept"] is not None
