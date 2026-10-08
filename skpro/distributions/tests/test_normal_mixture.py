# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""NormalMixture-specific tests not covered by generic distribution checks."""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions.normal_mixture import NormalMixture
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(NormalMixture),
    reason="run test only if tested object has changed",
)
def test_pi_is_normalized_per_row():
    """Mixture weights should be normalized row-wise at construction."""
    pi = np.array([[0.3, 0.7], [1.0, 2.0]])
    mu = np.array([[[0, 1], [3, 4]], [[1, 2], [4, 5]]])
    sigma = np.ones((2, 2, 2))

    d = NormalMixture(pi=pi, mu=mu, sigma=sigma)
    assert np.allclose(d._pi.sum(axis=1), 1.0)


@pytest.mark.skipif(
    not run_test_for_class(NormalMixture),
    reason="run test only if tested object has changed",
)
def test_mean_and_var_match_closed_form_scalar():
    """Mean/variance should match closed-form mixture formulas in scalar case."""
    pi = np.array([0.5, 0.5])
    mu = np.array([0.0, 2.0])
    sigma = np.array([1.0, 1.0])

    d = NormalMixture(pi=pi, mu=mu, sigma=sigma)

    # E[X] = sum_k pi_k mu_k
    expected_mean = 1.0
    # E[X^2] = sum_k pi_k (sigma_k^2 + mu_k^2) = 3, Var = E[X^2] - E[X]^2 = 2
    expected_var = 2.0

    assert np.isclose(d.mean(), expected_mean)
    assert np.isclose(d.var(), expected_var)


@pytest.mark.skipif(
    not run_test_for_class(NormalMixture),
    reason="run test only if tested object has changed",
)
def test_single_component_reduces_to_normal_pdf_and_cdf():
    """With one active component, pdf/cdf should match that Normal component."""
    pi = np.array([[1.0, 0.0]])
    mu = np.array([[[0.0], [3.0]]])
    sigma = np.array([[[1.0], [1.0]]])
    d = NormalMixture(pi=pi, mu=mu, sigma=sigma)

    x = np.array([[0.0]])
    pdf = d.pdf(x).iloc[0, 0]
    cdf = d.cdf(x).iloc[0, 0]

    assert np.isclose(pdf, 1.0 / np.sqrt(2.0 * np.pi))
    assert np.isclose(cdf, 0.5)


@pytest.mark.skipif(
    not run_test_for_class(NormalMixture),
    reason="run test only if tested object has changed",
)
def test_rowwise_weights_change_rowwise_mean():
    """Per-sample weights should produce different means per row."""
    pi = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([[[0.0], [10.0]], [[0.0], [10.0]]])
    sigma = np.ones((2, 2, 1))

    d = NormalMixture(pi=pi, mu=mu, sigma=sigma)
    means = d.mean().iloc[:, 0].to_numpy()

    assert np.isclose(means[0], 1.0)
    assert np.isclose(means[1], 9.0)


@pytest.mark.skipif(
    not run_test_for_class(NormalMixture),
    reason="run test only if tested object has changed",
)
def test_sampling_mean_matches_theoretical_mean():
    """Large-sample mean should approximate theoretical mixture mean."""
    pi = np.array([[0.5, 0.5]])
    mu = np.array([[[0.0], [4.0]]])
    sigma = np.array([[[0.5], [0.5]]])
    d = NormalMixture(pi=pi, mu=mu, sigma=sigma)

    # Use global seed consumed by distribution sampling path.
    np.random.seed(42)
    samples = d.sample(n_samples=10000)
    sample_mean = float(samples.values.mean())
    theoretical_mean = float(d.mean().iloc[0, 0])

    assert abs(sample_mean - theoretical_mean) < 0.1


@pytest.mark.skipif(
    not run_test_for_class(NormalMixture),
    reason="run test only if tested object has changed",
)
def test_energy_returns_non_negative_dataframe():
    """Energy outputs should be non-negative and keep expected tabular shape."""
    pi = np.array([[1.0, 0.0]])
    mu = np.array([[[0.0], [3.0]]])
    sigma = np.array([[[1.0], [1.0]]])
    d = NormalMixture(pi=pi, mu=mu, sigma=sigma)

    e_self = d.energy()
    e_x = d.energy(0.5)

    assert isinstance(e_self, pd.DataFrame)
    assert isinstance(e_x, pd.DataFrame)
    assert e_self.shape == (1, 1)
    assert e_x.shape == (1, 1)
    assert e_self.iloc[0, 0] >= 0
    assert e_x.iloc[0, 0] >= 0


def _energy_by_quadrature(pi, mu, sigma):
    """E|X - Y| for i.i.d. mixture draws, as 2 * integral of F (1 - F)."""
    from scipy import integrate, stats

    def cdf(x):
        return sum(p * stats.norm.cdf(x, m, s) for p, m, s in zip(pi, mu, sigma))

    value, _ = integrate.quad(lambda x: cdf(x) * (1 - cdf(x)), -np.inf, np.inf)
    return 2 * value


@pytest.mark.skipif(
    not run_test_for_class(NormalMixture),
    reason="run test only if tested object has changed",
)
@pytest.mark.parametrize(
    "pi, mu, sigma",
    [
        ([1.0], [0.0], [1.0]),
        ([0.4, 0.6], [0.0, 3.0], [1.0, 2.0]),
        ([0.5, 0.5], [-10.0, 10.0], [1.0, 1.0]),
        ([0.2, 0.3, 0.5], [-1.0, 0.5, 4.0], [0.5, 2.0, 1.0]),
    ],
)
def test_energy_matches_quadrature_scalar(pi, mu, sigma):
    """Energy of self should be E|X - Y|, including the component means."""
    d = NormalMixture(pi=np.array(pi), mu=np.array(mu), sigma=np.array(sigma))

    assert np.isclose(float(d.energy()), _energy_by_quadrature(pi, mu, sigma))


@pytest.mark.skipif(
    not run_test_for_class(NormalMixture),
    reason="run test only if tested object has changed",
)
def test_energy_single_component_is_normal_energy():
    """One component is a Normal, whose energy is 2 sigma / sqrt(pi)."""
    d = NormalMixture(pi=np.array([1.0]), mu=np.array([3.0]), sigma=np.array([2.5]))

    assert np.isclose(float(d.energy()), 2 * 2.5 / np.sqrt(np.pi))


@pytest.mark.skipif(
    not run_test_for_class(NormalMixture),
    reason="run test only if tested object has changed",
)
def test_energy_rowwise_weights_and_outputs():
    """Energy sums over outputs and uses each row's own weights and components."""
    pi = np.array([[0.3, 0.7], [0.5, 0.5]])
    mu = np.array([[[0, 1], [3, 4]], [[1, 2], [4, 5]]])
    sigma = np.array([[[1, 2], [1, 1]], [[0.5, 1], [2, 1]]])

    d = NormalMixture(pi=pi, mu=mu, sigma=sigma)

    expected = [
        sum(_energy_by_quadrature(pi[i], mu[i, :, j], sigma[i, :, j]) for j in range(2))
        for i in range(2)
    ]
    assert np.allclose(np.asarray(d.energy()).ravel(), expected)


@pytest.mark.skipif(
    not run_test_for_class(NormalMixture),
    reason="run test only if tested object has changed",
)
def test_crps_from_energy_matches_quadrature():
    """CRPS = E|X - y| - E|X - Y| / 2 should match its defining integral."""
    from scipy import integrate, stats

    pi, mu, sigma, y = [0.4, 0.6], [0.0, 3.0], [1.0, 2.0], 1.0
    d = NormalMixture(pi=np.array(pi), mu=np.array(mu), sigma=np.array(sigma))

    crps = float(d.energy(y)) - float(d.energy()) / 2

    def cdf(x):
        return sum(p * stats.norm.cdf(x, m, s) for p, m, s in zip(pi, mu, sigma))

    expected, _ = integrate.quad(
        lambda x: (cdf(x) - (x >= y)) ** 2, -40, 40, points=[y], limit=300
    )
    assert np.isclose(crps, expected)
