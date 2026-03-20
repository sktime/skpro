"""Linear closed-form and MCMC Bayesian regressor tests."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from skpro.distributions.base import BaseDistribution
from skpro.regression.bayesian import BayesianLinearClosedFormRegressor

_PYMC_EXTRAS_AVAILABLE = _check_soft_dependencies("pymc_extras", severity="none")


def _make_data(n=30, n_features=2, seed=0, col_prefix="x", index_start=0):
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
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y_vals = intercept + slope * x + rng.standard_normal(n) * noise_std
    X = pd.DataFrame({"x": x})
    y = pd.DataFrame({"y": y_vals})
    return X, y


class TestClosedFormFitPredict:
    @pytest.fixture(autouse=True)
    def _setup(self):
        X, y = _make_data(n=40, n_features=3, seed=1)
        X_te, _ = _make_data(n=10, n_features=3, seed=2, index_start=40)
        self.X, self.y, self.X_te = X, y, X_te
        self.reg = BayesianLinearClosedFormRegressor()
        self.reg.fit(X, y)

    def test_is_fitted(self):
        assert self.reg.is_fitted

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

    def test_predict_var_positive(self):
        pv = self.reg.predict_var(self.X_te)
        assert isinstance(pv, pd.DataFrame)
        assert (pv.values > 0).all()

    def test_predict_var_index_matches(self):
        pv = self.reg.predict_var(self.X_te)
        assert list(pv.index) == list(self.X_te.index)


class TestClosedFormNonDefaultIndex:
    def test_non_sequential_integer_index_preserved(self):
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


class TestPosteriorStructure:
    @pytest.fixture(autouse=True)
    def _setup(self):
        X, y = _make_data(n=30, n_features=2, seed=5)
        self.reg = BayesianLinearClosedFormRegressor(fit_intercept=True)
        self.reg.fit(X, y)
        self.n_coef = X.shape[1] + 1

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


class TestPosteriorAccessors:
    @pytest.mark.skipif(not _check_soft_dependencies("arviz", severity="none"), reason="arviz not installed")
    def test_posterior_summary_variational_and_mcmc(self):
        from skpro.regression.bayesian import BaseBayesianRegressor

        class DummyApprox:
            def sample(self, draws):
                import pandas as pd

                return pd.DataFrame({"a": [0, 1], "b": [2, 3]})

        reg = BaseBayesianRegressor()
        reg.approx_ = DummyApprox()
        with pytest.raises(ValueError, match="Can only convert xarray dataarray"):
            reg._get_posterior_summary_from_posterior()

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


class TestAnalyticalCorrectness:
    def test_recovers_slope_for_simple_linear_data(self):
        slope, intercept = 2.5, -0.5
        X, y = _make_linear_data(
            slope=slope, intercept=intercept, n=200, noise_std=0.02, seed=1
        )
        reg = BayesianLinearClosedFormRegressor(
            prior_mean=0.0,
            prior_precision=0.001,
            noise_precision=2500.0,
            fit_intercept=True,
        )
        reg.fit(X, y)
        est_intercept = reg.posterior_mu_[0, 0]
        est_slope = reg.posterior_mu_[1, 0]
        assert abs(est_slope - slope) < 0.1
        assert abs(est_intercept - intercept) < 0.1

    def test_no_intercept_slope_recovery(self):
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
        X = pd.DataFrame({"x": [100.0]})
        y = pd.DataFrame({"y": [100.0]})
        prior_mean = 5.0
        reg = BayesianLinearClosedFormRegressor(
            prior_mean=prior_mean,
            prior_precision=1e6,
            noise_precision=0.001,
            fit_intercept=False,
        )
        reg.fit(X, y)
        assert abs(reg.posterior_mu_[0, 0] - prior_mean) < 0.5

    def test_data_dominates_with_weak_prior(self):
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
        assert var_large < var_small


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
        X, y = _make_data(n=20, seed=3)
        reg = BayesianLinearClosedFormRegressor()
        reg.fit(X, y)
        p1 = reg.predict_proba(X).mean()
        p2 = reg.predict_proba(X).mean()
        pd.testing.assert_frame_equal(p1, p2)

    def test_predict_proba_different_on_ood_x(self):
        X_tr = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        y_tr = pd.DataFrame({"y": [0.0, 2.0, 4.0]})
        reg = BayesianLinearClosedFormRegressor(
            prior_precision=0.01, noise_precision=1000.0, fit_intercept=False
        )
        reg.fit(X_tr, y_tr)
        X_far = pd.DataFrame({"x": [100.0]})
        pred_mean = reg.predict(X_far).iloc[0, 0]
        train_mean = y_tr["y"].mean()
        assert abs(pred_mean - train_mean) > 50


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
        reg = self._make_reg(sampler_config={"cores": 3, "nuts_sampler": "numpyro"})
        assert reg.sample_kwargs.get("cores") == 3
        assert reg.sample_kwargs.get("nuts_sampler") == "numpyro"

    def test_extra_sampler_keys_not_in_base_params(self):
        reg = self._make_reg(sampler_config={"cores": 3})
        assert not hasattr(reg, "cores")

    def test_default_sampler_params_complete(self):
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
        assert reg.prior_config["intercept"] is not None
