"""Tests for _normalize_dist_str and its integration into NGBoostRegressor.

These tests cover:
- Unit tests for ``_normalize_dist_str`` (no external dependencies needed)
- Cross-regressor alias consistency checks
- Integration tests for NGBoostRegressor (skipped if ngboost not installed)
"""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["paramsureliya"]

import warnings

import pytest

# ---------------------------------------------------------------------------
# Unit tests for _normalize_dist_str
# ---------------------------------------------------------------------------


class TestNormalizeDistStr:
    """Unit tests for the standalone _normalize_dist_str utility."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from skpro.regression._dist_utils import _normalize_dist_str

        self.fn = _normalize_dist_str

    @pytest.mark.parametrize(
        "canonical",
        [
            "Normal",
            "Laplace",
            "LogNormal",
            "TDistribution",
            "Poisson",
            "Exponential",
            "Gamma",
            "Beta",
            "Weibull",
            "Cauchy",
            "Binomial",
            "NegativeBinomial",
            "InverseGaussian",
            "Tweedie",
        ],
    )
    def test_canonical_passthrough(self, canonical):
        """Already-canonical names must be returned as-is."""
        assert self.fn(canonical) == canonical

    @pytest.mark.parametrize(
        "alias",
        ["normal", "gaussian", "Gaussian", "GAUSSIAN", "norm", "NORMAL"],
    )
    def test_normal_aliases(self, alias):
        """All common spellings of Normal/Gaussian map to 'Normal'."""
        assert self.fn(alias) == "Normal"

    @pytest.mark.parametrize(
        "alias, expected",
        [
            # Laplace
            ("laplace", "Laplace"),
            ("double_exponential", "Laplace"),
            # LogNormal
            ("lognormal", "LogNormal"),
            ("log_normal", "LogNormal"),
            ("log-normal", "LogNormal"),
            ("log normal", "LogNormal"),
            # TDistribution — plain "t" is used by ResidualDouble
            ("t", "TDistribution"),
            ("tdistribution", "TDistribution"),
            ("t_distribution", "TDistribution"),
            ("student_t", "TDistribution"),
            ("studentt", "TDistribution"),
            # Poisson
            ("poisson", "Poisson"),
            # Exponential
            ("exponential", "Exponential"),
            ("exp", "Exponential"),
            # Gamma
            ("gamma", "Gamma"),
            # Beta
            ("beta", "Beta"),
            # Weibull
            ("weibull", "Weibull"),
            # Cauchy
            ("cauchy", "Cauchy"),
            # Binomial
            ("binomial", "Binomial"),
            ("binom", "Binomial"),
            # NegativeBinomial — GlumRegressor uses dot-notation
            ("negative.binomial", "NegativeBinomial"),
            ("negative_binomial", "NegativeBinomial"),
            ("negativebinomial", "NegativeBinomial"),
            ("negbin", "NegativeBinomial"),
            # InverseGaussian — GlumRegressor uses dot-notation
            ("inverse.gaussian", "InverseGaussian"),
            ("inverse_gaussian", "InverseGaussian"),
            ("inversegaussian", "InverseGaussian"),
            # Tweedie
            ("tweedie", "Tweedie"),
        ],
    )
    def test_alias_mapping(self, alias, expected):
        assert (
            self.fn(alias) == expected
        ), f"_normalize_dist_str({alias!r}) should return {expected!r}"

    def test_non_string_passthrough(self):
        """Non-string inputs (classes, objects, None) must pass through unchanged."""

        class FakeDist:
            pass

        obj = FakeDist()
        assert self.fn(obj) is obj
        assert self.fn(None) is None
        assert self.fn(42) == 42

    def test_unknown_string_warns_not_raises(self):
        """An unrecognised string emits UserWarning but does not raise."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.fn("totally_unknown_dist_xyz")
        assert result == "totally_unknown_dist_xyz"
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "totally_unknown_dist_xyz" in str(w[0].message)

    @pytest.mark.parametrize("alias", ["gaussian", "lognormal", "t", "Normal"])
    def test_idempotent(self, alias):
        """Calling the function twice must give the same result as once."""
        once = self.fn(alias)
        twice = self.fn(once)
        assert once == twice, (
            f"_normalize_dist_str is not idempotent for {alias!r}: "
            f"first call -> {once!r}, second call -> {twice!r}"
        )


# ---------------------------------------------------------------------------
# Cross-regressor alias consistency (no external deps)
# ---------------------------------------------------------------------------


class TestCrossRegressorAliasConsistency:
    """The same alias must resolve to the same canonical name regardless of
    which regressor calls _normalize_dist_str — this is the invariant that
    makes GridSearchCV across regressors work.
    """

    def test_gaussian_always_means_normal(self):
        from skpro.regression._dist_utils import _normalize_dist_str

        for alias in ("gaussian", "Gaussian", "GAUSSIAN", "normal", "Normal"):
            assert (
                _normalize_dist_str(alias) == "Normal"
            ), f"Expected 'Normal' for alias {alias!r}"

    def test_t_alias_consistent(self):
        """'t' (used by ResidualDouble) must map to 'TDistribution'."""
        from skpro.regression._dist_utils import _normalize_dist_str

        for alias in ("t", "TDistribution", "tdistribution", "t_distribution"):
            assert (
                _normalize_dist_str(alias) == "TDistribution"
            ), f"Expected 'TDistribution' for alias {alias!r}"

    def test_glum_dot_notation_normalized(self):
        """GlumRegressor uses dot-notation for two distributions."""
        from skpro.regression._dist_utils import _normalize_dist_str

        assert _normalize_dist_str("negative.binomial") == "NegativeBinomial"
        assert _normalize_dist_str("inverse.gaussian") == "InverseGaussian"


# ---------------------------------------------------------------------------
# Integration tests — NGBoostRegressor accepts aliases end-to-end
# (skipped automatically if ngboost is not installed)
# ---------------------------------------------------------------------------


class TestNGBoostRegressorAliases:
    """Verify that NGBoostRegressor accepts distribution string aliases.

    Skipped automatically if ngboost is not importable. To run locally::

        pip install ngboost
        pytest skpro/regression/tests/test_dist_utils.py -v
    """

    @pytest.fixture(autouse=True)
    def _require_ngboost(self):
        pytest.importorskip("ngboost", reason="ngboost not installed")

    @pytest.fixture
    def xy(self):
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        X, y = load_diabetes(return_X_y=True, as_frame=True)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.1, random_state=0)
        return X_tr, X_te, y_tr

    @pytest.mark.parametrize(
        "alias",
        [
            "Normal",  # canonical — must still work
            "normal",  # lowercase
            "gaussian",  # common synonym
            "Gaussian",  # mixed-case synonym
            "Laplace",
            "laplace",
            "Poisson",
            "poisson",
            "LogNormal",
            "lognormal",
            "log_normal",
        ],
    )
    def test_fit_predict_with_alias(self, alias, xy):
        """NGBoostRegressor(dist=alias) must fit and predict_proba without error."""
        from skpro.regression.ensemble import NGBoostRegressor

        X_tr, X_te, y_tr = xy
        est = NGBoostRegressor(dist=alias, n_estimators=20, verbose=False)
        est.fit(X_tr, y_tr)
        y_pred = est.predict_proba(X_te)
        assert y_pred is not None

    @pytest.mark.parametrize(
        "alias_a, alias_b",
        [
            ("Normal", "gaussian"),
            ("Normal", "normal"),
            ("LogNormal", "lognormal"),
            ("Laplace", "laplace"),
        ],
    )
    def test_aliases_produce_same_distribution_type(self, alias_a, alias_b, xy):
        """Two aliases for the same distribution must return the same output type."""
        from skpro.regression.ensemble import NGBoostRegressor

        X_tr, X_te, y_tr = xy
        est_a = NGBoostRegressor(dist=alias_a, n_estimators=20, verbose=False)
        est_b = NGBoostRegressor(dist=alias_b, n_estimators=20, verbose=False)
        est_a.fit(X_tr, y_tr)
        est_b.fit(X_tr, y_tr)

        pred_a = est_a.predict_proba(X_te)
        pred_b = est_b.predict_proba(X_te)

        assert type(pred_a) is type(pred_b), (
            f"dist={alias_a!r} -> {type(pred_a).__name__}, "
            f"dist={alias_b!r} -> {type(pred_b).__name__} — expected the same type"
        )
