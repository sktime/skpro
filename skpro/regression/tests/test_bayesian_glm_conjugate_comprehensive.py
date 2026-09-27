"""Comprehensive tests for BayesianConjugateGLMRegressor.

Covers:
- Posterior math correctness (Bishop PRML Ch. 3)
- Student-t predictive path (with noise priors)
- Online update (_update)
- Edge cases: dimension mismatch, missing g, ARD
- Numerical stability
- log_marginal_likelihood
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from skpro.regression.bayesian._glm_conjugate import BayesianConjugateGLMRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_Xy(n=30, p=2, add_const=True, rng=None):
    """Return a simple (X, y) pair for testing."""
    rng = rng or RNG
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    y = pd.DataFrame(rng.standard_normal(n), columns=["target"])
    return X, y


def _make_est(n_features=2, add_constant=True, **kwargs):
    """Return a minimal BayesianConjugateGLMRegressor with identity prior."""
    n_coefs = n_features + int(add_constant)
    defaults = dict(
        coefs_prior_cov=np.eye(n_coefs),
        coefs_prior_mu=np.zeros((n_coefs, 1)),
        noise_precision=1.0,
        add_constant=add_constant,
    )
    defaults.update(kwargs)
    return BayesianConjugateGLMRegressor(**defaults)


# ---------------------------------------------------------------------------
# 1. Posterior math correctness (Bishop PRML eq. 3.50-3.51)
# ---------------------------------------------------------------------------


def test_posterior_mean_formula():
    """Posterior mean must equal closed-form Bishop eq. 3.50."""
    X, y = _make_Xy(n=30, p=2, add_const=True)
    n_coefs = 3
    S0 = np.eye(n_coefs) * 2.0
    m0 = np.ones((n_coefs, 1)) * 0.5
    tau = 1.5

    est = BayesianConjugateGLMRegressor(
        coefs_prior_cov=S0,
        coefs_prior_mu=m0,
        noise_precision=tau,
        add_constant=True,
    )
    est.fit(X, y)

    # Re-derive manually
    X_arr = np.column_stack([np.ones(len(X)), X.values])
    y_arr = y.values
    S0_inv = np.linalg.inv(S0)
    SN_inv = S0_inv + tau * X_arr.T @ X_arr
    SN = np.linalg.inv(SN_inv)
    mN = SN @ (S0_inv @ m0 + tau * X_arr.T @ y_arr)

    np.testing.assert_allclose(
        est._coefs_posterior_mu,
        mN,
        rtol=1e-8,
        atol=1e-10,
        err_msg="Posterior mean does not match Bishop eq. 3.50",
    )


def test_posterior_covariance_formula():
    """Posterior covariance must equal closed-form Bishop eq. 3.51."""
    X, y = _make_Xy(n=30, p=2, add_const=True)
    n_coefs = 3
    S0 = np.eye(n_coefs)
    tau = 2.0

    est = _make_est(n_features=2, add_constant=True)
    est.noise_precision = tau
    est.coefs_prior_cov = S0
    est.fit(X, y)

    X_arr = np.column_stack([np.ones(len(X)), X.values])
    SN_inv = np.linalg.inv(S0) + tau * X_arr.T @ X_arr
    SN_expected = np.linalg.inv(SN_inv)

    np.testing.assert_allclose(
        est._coefs_posterior_cov,
        SN_expected,
        rtol=1e-8,
        atol=1e-10,
        err_msg="Posterior covariance does not match Bishop eq. 3.51",
    )


def test_noise_posterior_shape_update():
    """aN = a0 + N/2 per Normal-Gamma conjugacy."""
    X, y = _make_Xy(n=25, p=2)
    a0, b0 = 2.0, 3.0
    est = _make_est(
        n_features=2,
        noise_prior_shape=a0,
        noise_prior_rate=b0,
    )
    est.fit(X, y)

    expected_aN = a0 + len(X) / 2
    assert est._noise_posterior_shape == pytest.approx(expected_aN, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Predictive distribution types
# ---------------------------------------------------------------------------


def test_predict_proba_returns_normal_without_noise_priors():
    """Without noise priors, predict_proba must return a Normal distribution."""
    from skpro.distributions.normal import Normal

    X, y = _make_Xy()
    est = _make_est()
    est.fit(X, y)
    dist = est.predict_proba(X)
    assert isinstance(dist, Normal), f"Expected Normal, got {type(dist)}"


def test_predict_proba_returns_tdistribution_with_noise_priors():
    """With noise priors, predict_proba must return a TDistribution."""
    from skpro.distributions.t import TDistribution

    X, y = _make_Xy()
    est = _make_est(noise_prior_shape=2.0, noise_prior_rate=2.0)
    est.fit(X, y)
    dist = est.predict_proba(X)
    assert isinstance(dist, TDistribution), f"Expected TDistribution, got {type(dist)}"


def test_tdistribution_degrees_of_freedom():
    """TDistribution df must equal 2 * aN."""
    from skpro.distributions.t import TDistribution

    X, y = _make_Xy(n=20)
    a0 = 3.0
    est = _make_est(noise_prior_shape=a0, noise_prior_rate=2.0)
    est.fit(X, y)
    dist = est.predict_proba(X)
    assert isinstance(dist, TDistribution)

    expected_df = 2 * est._noise_posterior_shape
    # df is stored as a scalar or array — extract representative value
    df_val = np.asarray(dist.df).flat[0]
    assert df_val == pytest.approx(expected_df, rel=1e-9)


def test_predict_proba_mean_matches_posterior_mean():
    """Predictive mean must equal X @ coefs_posterior_mu."""
    X, y = _make_Xy(n=30, p=2)
    est = _make_est()
    est.fit(X, y)

    pred_mean = est.predict_proba(X).mean()
    X_arr = np.column_stack([np.ones(len(X)), X.values])
    expected = pd.DataFrame(
        X_arr @ est._coefs_posterior_mu,
        index=X.index,
        columns=["target"],
    )
    np.testing.assert_allclose(
        pred_mean.values,
        expected.values,
        rtol=1e-7,
        atol=1e-9,
        err_msg="Predictive mean does not equal X @ posterior_mu",
    )


def test_predict_proba_finite_outputs():
    """pdf and log_pdf must be finite for all predictions."""
    X, y = _make_Xy(n=20)
    est = _make_est()
    est.fit(X, y)
    dist = est.predict_proba(X)
    x_test = pd.DataFrame({"target": np.ones(len(X))}, index=X.index)
    assert np.isfinite(dist.pdf(x_test).values).all()
    assert np.isfinite(dist.log_pdf(x_test).values).all()


# ---------------------------------------------------------------------------
# 3. Online update
# ---------------------------------------------------------------------------


def test_online_update_shifts_posterior_toward_data():
    """After update with strong signal, posterior mean should change."""
    X1, y1 = _make_Xy(n=20, p=2)
    X2, y2 = _make_Xy(n=20, p=2)

    est = _make_est()
    est.fit(X1, y1)
    mu_before = est._coefs_posterior_mu.copy()

    est._update(X2, y2)
    mu_after = est._coefs_posterior_mu

    # Posterior must have changed
    assert not np.allclose(
        mu_before, mu_after
    ), "Posterior mean did not change after _update"


def test_batch_vs_sequential_update_equivalence():
    """Fitting on X1+X2 at once equals fit(X1) then update(X2)."""
    rng = np.random.default_rng(7)
    X1, y1 = _make_Xy(n=15, p=2, rng=rng)
    X2, y2 = _make_Xy(n=15, p=2, rng=rng)

    # Sequential
    est_seq = _make_est()
    est_seq.fit(X1, y1)
    est_seq._update(X2, y2)

    # Batch
    X_all = pd.concat([X1, X2], ignore_index=True)
    y_all = pd.concat([y1, y2], ignore_index=True)
    est_batch = _make_est()
    est_batch.fit(X_all, y_all)

    np.testing.assert_allclose(
        est_seq._coefs_posterior_mu,
        est_batch._coefs_posterior_mu,
        rtol=1e-7,
        atol=1e-9,
        err_msg="Sequential update != batch fit (posterior mean mismatch)",
    )
    np.testing.assert_allclose(
        est_seq._coefs_posterior_cov,
        est_batch._coefs_posterior_cov,
        rtol=1e-7,
        atol=1e-9,
        err_msg="Sequential update != batch fit (posterior cov mismatch)",
    )


# ---------------------------------------------------------------------------
# 4. Edge cases and input validation
# ---------------------------------------------------------------------------


def test_add_constant_false():
    """add_constant=False: no intercept column added, shapes must be consistent."""
    X, y = _make_Xy(n=20, p=3)
    est = _make_est(n_features=3, add_constant=False)
    est.fit(X, y)
    dist = est.predict_proba(X)
    assert dist.mean().shape == (len(X), 1)


def test_gprior_requires_g_parameter():
    """prior_type='gprior' without g must raise ValueError."""
    X, y = _make_Xy()
    est = BayesianConjugateGLMRegressor(
        coefs_prior_cov=np.eye(3),
        coefs_prior_mu=np.zeros((3, 1)),
        noise_precision=1.0,
        prior_type="gprior",
        g=None,
    )
    with pytest.raises(ValueError, match="g"):
        est.fit(X, y)


def test_ard_mode():
    """ARD mode: ard=True with ard_lambda produces valid predictions."""
    X, y = _make_Xy(n=20, p=2)
    n_coefs = 3  # 2 features + intercept
    est = BayesianConjugateGLMRegressor(
        ard=True,
        ard_lambda=np.ones(n_coefs),
        noise_precision=1.0,
        add_constant=True,
    )
    est.fit(X, y)
    dist = est.predict_proba(X)
    assert dist.mean().shape == (len(X), 1)


def test_dimension_mismatch_raises():
    """coefs_prior_mu and coefs_prior_cov size mismatch must raise ValueError."""
    X, y = _make_Xy()
    est = BayesianConjugateGLMRegressor(
        coefs_prior_cov=np.eye(3),
        coefs_prior_mu=np.zeros((5, 1)),  # wrong size
        noise_precision=1.0,
        add_constant=True,
    )
    with pytest.raises(ValueError, match="Dimensionality"):
        est.fit(X, y)


def test_prior_via_precision_matrix():
    """Specifying coefs_prior_precision directly must give same result as via cov."""
    X, y = _make_Xy(n=25, p=2)
    S0 = np.eye(3) * 3.0
    S0_inv = np.linalg.inv(S0)
    m0 = np.zeros((3, 1))

    est_cov = BayesianConjugateGLMRegressor(
        coefs_prior_cov=S0,
        coefs_prior_mu=m0,
        noise_precision=1.0,
        add_constant=True,
    )
    est_prec = BayesianConjugateGLMRegressor(
        coefs_prior_precision=S0_inv,
        coefs_prior_mu=m0,
        noise_precision=1.0,
        add_constant=True,
    )
    est_cov.fit(X, y)
    est_prec.fit(X, y)

    np.testing.assert_allclose(
        est_cov._coefs_posterior_mu,
        est_prec._coefs_posterior_mu,
        rtol=1e-8,
        atol=1e-10,
        err_msg="Prior via cov and via precision must yield same posterior",
    )


# ---------------------------------------------------------------------------
# 5. Numerical stability
# ---------------------------------------------------------------------------


def test_large_dataset_finite_outputs():
    """N=500 should produce finite posterior and finite predictive pdf."""
    X, y = _make_Xy(n=500, p=4)
    est = _make_est(n_features=4)
    est.fit(X, y)
    assert np.isfinite(est._coefs_posterior_mu).all()
    assert np.isfinite(est._coefs_posterior_cov).all()

    dist = est.predict_proba(X.iloc[:10])
    x_test = pd.DataFrame({"target": np.ones(10)}, index=X.index[:10])
    assert np.isfinite(dist.pdf(x_test).values).all()


def test_small_noise_precision_finite():
    """Very small noise_precision (near-uninformative) must not produce NaN."""
    X, y = _make_Xy(n=20, p=2)
    est = _make_est(n_features=2)
    est.noise_precision = 1e-8
    # Re-create with tiny precision
    est2 = BayesianConjugateGLMRegressor(
        coefs_prior_cov=np.eye(3) * 10,
        coefs_prior_mu=np.zeros((3, 1)),
        noise_precision=1e-8,
        add_constant=True,
    )
    est2.fit(X, y)
    dist = est2.predict_proba(X)
    assert np.isfinite(dist.mean().values).all()


# ---------------------------------------------------------------------------
# 6. log_marginal_likelihood
# ---------------------------------------------------------------------------


def test_log_marginal_likelihood_is_finite():
    """log_marginal_likelihood must return a finite scalar."""
    X, y = _make_Xy(n=20, p=2)
    est = _make_est()
    est.fit(X, y)
    lml = est.log_marginal_likelihood(X, y)
    assert np.isfinite(lml), f"log_marginal_likelihood is not finite: {lml}"


def test_log_marginal_likelihood_model_comparison():
    """True generative model should have higher evidence than random model.

    We generate y from a known beta, then compare LML of correct prior
    centred on true beta vs a very wrong prior.
    """
    rng = np.random.default_rng(99)
    n, p = 50, 2
    true_beta = np.array([[1.0], [2.0], [-1.0]])  # intercept + 2 weights
    X_raw = rng.standard_normal((n, p))
    X_aug = np.column_stack([np.ones(n), X_raw])
    y_raw = X_aug @ true_beta + rng.standard_normal((n, 1)) * 0.3

    X = pd.DataFrame(X_raw, columns=["x0", "x1"])
    y = pd.DataFrame(y_raw, columns=["target"])

    # Good model: prior centred on true beta
    good_est = BayesianConjugateGLMRegressor(
        coefs_prior_cov=np.eye(3) * 0.5,
        coefs_prior_mu=true_beta,
        noise_precision=1.0,
        add_constant=True,
    )
    good_est.fit(X, y)

    # Bad model: prior far from truth
    bad_est = BayesianConjugateGLMRegressor(
        coefs_prior_cov=np.eye(3) * 0.5,
        coefs_prior_mu=np.full((3, 1), 100.0),
        noise_precision=1.0,
        add_constant=True,
    )
    bad_est.fit(X, y)

    lml_good = good_est.log_marginal_likelihood(X, y)
    lml_bad = bad_est.log_marginal_likelihood(X, y)

    assert (
        lml_good > lml_bad
    ), f"Good model LML ({lml_good:.2f}) <= bad model LML ({lml_bad:.2f})"
