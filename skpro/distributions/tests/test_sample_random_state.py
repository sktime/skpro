# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for random_state parameter in BaseDistribution.sample, resolves #661."""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions.empirical import Empirical
from skpro.distributions.normal import Normal


def _make_normal():
    """Helper: simple 2x1 Normal distribution."""
    index = pd.Index([0, 1])
    columns = pd.Index(["a"])
    return Normal(
        mu=pd.DataFrame([[0.0], [0.0]], index=index, columns=columns),
        sigma=pd.DataFrame([[1.0], [1.0]], index=index, columns=columns),
        index=index,
        columns=columns,
    )


def _make_empirical():
    """Helper: simple scalar Empirical distribution."""
    return Empirical(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))


# --- Test 1: same seed → same result (Normal) ---
def test_sample_random_state_reproducible_normal():
    """Same int seed produces identical samples."""
    dist = _make_normal()
    s1 = dist.sample(n_samples=5, random_state=42)
    s2 = dist.sample(n_samples=5, random_state=42)
    pd.testing.assert_frame_equal(s1, s2)


# --- Test 2: different seeds → different results ---
def test_sample_random_state_different_seeds():
    """Different seeds produce different samples (with overwhelming probability)."""
    dist = _make_normal()
    s1 = dist.sample(n_samples=10, random_state=42)
    s2 = dist.sample(n_samples=10, random_state=99)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(s1, s2)


# --- Test 3: np.random.Generator input works ---
def test_sample_random_state_generator_object():
    """np.random.Generator is accepted directly."""
    dist = _make_normal()
    rng = np.random.default_rng(42)
    result = dist.sample(n_samples=5, random_state=rng)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5 * len(dist.index)


# --- Test 4: None does not raise ---
def test_sample_random_state_none():
    """random_state=None works (uses global numpy state)."""
    dist = _make_normal()
    result = dist.sample(n_samples=3, random_state=None)
    assert isinstance(result, pd.DataFrame)


# --- Test 5: Empirical distribution works ---
def test_sample_random_state_empirical():
    """Empirical distribution respects random_state."""
    dist = _make_empirical()
    s1 = dist.sample(n_samples=10, random_state=7)
    s2 = dist.sample(n_samples=10, random_state=7)
    pd.testing.assert_frame_equal(s1, s2)


# --- Test 6: backward compat — old API dist.sample() still works ---
def test_sample_backward_compat_no_random_state():
    """Calling sample() without random_state still works (backward compat)."""
    dist = _make_normal()
    result = dist.sample(n_samples=3)
    assert isinstance(result, pd.DataFrame)


# --- Test 7: legacy np.random.RandomState is handled ---
def test_sample_random_state_legacy_randomstate():
    """Legacy np.random.RandomState is accepted."""
    dist = _make_normal()
    legacy_rng = np.random.RandomState(42)
    result = dist.sample(n_samples=5, random_state=legacy_rng)
    assert isinstance(result, pd.DataFrame)


# --- Test 8: invalid input raises ValueError ---
def test_sample_random_state_invalid():
    """Invalid random_state raises ValueError."""
    dist = _make_normal()
    with pytest.raises(ValueError, match="random_state must be"):
        dist.sample(random_state="bad_input")


# --- Test 9: context manager cleans up _sample_rng ---
def test_sample_rng_not_leaked():
    """_sample_rng is not left on the instance after sample() returns."""
    dist = _make_normal()
    assert not hasattr(dist, "_sample_rng")
    dist.sample(n_samples=3, random_state=42)
    assert not hasattr(dist, "_sample_rng")