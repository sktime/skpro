"""Base and interface tests for Bayesian regressors."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from skpro.distributions.base import BaseDistribution
from skpro.regression.base import BaseProbaRegressor
from skpro.regression.bayesian import BaseBayesianRegressor

_PYMC_EXTRAS_AVAILABLE = _check_soft_dependencies("pymc_extras", severity="none")

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stubs used for unit-testing hook dispatch WITHOUT PyMC
# ---------------------------------------------------------------------------


class _FakePredictions(dict):
    @property
    def data_vars(self):
        return self


class _FakeTrace:
    def __init__(self, predictions):
        self.predictions = predictions

    def groups(self):
        return ["predictions"]


class _MinimalNonMCRegressor(BaseBayesianRegressor):
    def _fit_posterior(self, X, y):
        self._y_columns = y.columns.tolist()
        self._mu_train = y.mean().values

    def _predict_proba_from_posterior(self, X):
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


class TestImportsAndHierarchy:
    def test_all_public_names_importable(self):
        import skpro.regression.bayesian as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"'{name}' listed in __all__ but not importable"

    def test_base_class_is_proba_regressor(self):
        assert issubclass(BaseBayesianRegressor, BaseProbaRegressor)

    def test_closed_form_is_bayesian_regressor(self):
        from skpro.regression.bayesian import BayesianLinearClosedFormRegressor

        assert issubclass(BayesianLinearClosedFormRegressor, BaseBayesianRegressor)

    def test_object_type_tag(self):
        assert BaseBayesianRegressor.get_class_tag("object_type") == "regressor_proba"


class TestBaseBayesianRegressorConstructor:
    def test_default_priors_and_robust(self):
        X, y = _make_data(n=10, n_features=3)
        reg = BaseBayesianRegressor()
        priors = reg._get_default_priors(X, y)
        assert "intercept" in priors and "slopes" in priors and "noise" in priors
        reg_robust = BaseBayesianRegressor(robust=True)
        priors_robust = reg_robust._get_default_priors(X, y)
        assert priors_robust["intercept"]["dist"] == "StudentT"
        assert priors_robust["slopes"]["dist"] == "StudentT"

    def test_prior_strength_scales_defaults(self):
        X, y = _make_data(n=10, n_features=2)
        reg = BaseBayesianRegressor(prior_strength=4.0)
        priors = reg._get_default_priors(X, y)
        assert priors["intercept"]["sd"] < 10.0 * y.values.std()

    def test_apply_prior_config_parsing(self):
        reg = BaseBayesianRegressor()
        model_vars = {
            "intercept": type(
                "Dummy",
                (),
                {"set_prior": lambda self, spec: setattr(self, "prior", spec)},
            )()
        }
        prior_cfg = {"intercept": "Normal(0,10)"}
        reg._apply_prior_config(model_vars, prior_cfg)
        assert hasattr(model_vars["intercept"], "prior")
        assert model_vars["intercept"].prior["dist"] == "Normal"

    @pytest.mark.skipif(
        not _check_soft_dependencies("pymc", severity="none"),
        reason="pymc not installed",
    )
    def test_variational_inference_stub(self):
        reg = BaseBayesianRegressor(inference_strategy="variational")
        with pytest.raises(NotImplementedError):
            reg._fit_variational_posterior(_make_data()[0], _make_data()[1])

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
        reg = BaseBayesianRegressor()
        X, y = _make_data(n=5)
        with pytest.raises(NotImplementedError):
            reg._build_model(X, y)


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
        reg = BaseBayesianRegressor()
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


class TestPosteriorSummaryFallback:
    @pytest.mark.skipif(
        not _check_soft_dependencies("arviz", severity="none"),
        reason="arviz not installed",
    )
    def test_raises_not_implemented_without_trace(self):
        reg = BaseBayesianRegressor()
        with pytest.raises(NotImplementedError, match="No posterior available"):
            reg.get_posterior_summary()


class TestNonMCHookContract:
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
        *_, pred = self._fit_predict()
        assert pred is not None


class TestGetTestParams:
    @pytest.mark.skipif(
        not _check_soft_dependencies("pymc", severity="none"),
        reason="pymc not installed",
    )
    def test_all_param_sets_create_valid_instances(self):
        for params in BaseBayesianRegressor.get_test_params():
            reg = BaseBayesianRegressor(**params)
            X, y = _make_data(n=10, n_features=2)
            if hasattr(reg, "inference_strategy") and reg.inference_strategy in [
                "conjugate",
                "variational",
                "mcmc",
            ]:
                with pytest.raises(NotImplementedError):
                    reg.fit(X, y)
            else:
                reg.fit(X, y)
