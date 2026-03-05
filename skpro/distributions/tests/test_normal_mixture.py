# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""NormalMixture-specific tests not covered by generic distribution checks."""

import numpy as np
import pandas as pd

from skpro.distributions.normal_mixture import NormalMixture


def test_pi_is_normalized_per_row():
    """Mixture weights should be normalized row-wise at construction."""
    pi = np.array([[0.3, 0.7], [1.0, 2.0]])
    mu = np.array([[[0, 1], [3, 4]], [[1, 2], [4, 5]]])
    sigma = np.ones((2, 2, 2))

    d = NormalMixture(pi=pi, mu=mu, sigma=sigma)
    assert np.allclose(d._pi.sum(axis=1), 1.0)


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


def test_rowwise_weights_change_rowwise_mean():
    """Per-sample weights should produce different means per row."""
    pi = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([[[0.0], [10.0]], [[0.0], [10.0]]])
    sigma = np.ones((2, 2, 1))

    d = NormalMixture(pi=pi, mu=mu, sigma=sigma)
    means = d.mean().iloc[:, 0].to_numpy()

    assert np.isclose(means[0], 1.0)
    assert np.isclose(means[1], 9.0)


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
