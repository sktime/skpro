# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for TemporalNormal distribution."""

__author__ = ["arnavk23"]

import numpy as np
import pandas as pd
import pytest

from skpro.distributions.temporal_normal import TemporalNormal


def test_temporal_normal_basic():
    """Test basic functionality of TemporalNormal distribution."""
    # Test with time-varying mean and constant variance
    mu_t = pd.Series([0, 1, 2, 3, 4], index=pd.RangeIndex(5))
    dist = TemporalNormal(mu=mu_t, sigma=1.0)
    
    assert dist.shape == (5, 1)
    assert len(dist) == 5
    
    # Test mean and variance methods
    mean = dist.mean()
    var = dist.var()
    
    assert mean.shape == (5, 1)
    assert var.shape == (5, 1)
    
    # Verify mean matches input
    np.testing.assert_allclose(mean.flatten(), mu_t.values, rtol=1e-10)


def test_temporal_normal_time_varying_both():
    """Test TemporalNormal with both mean and variance varying over time."""
    mu_t = pd.Series([0, 1, 2, 3, 4], index=pd.RangeIndex(5))
    sigma_t = pd.Series([0.5, 0.7, 1.0, 1.2, 1.5], index=pd.RangeIndex(5))
    dist = TemporalNormal(mu=mu_t, sigma=sigma_t)
    
    assert dist.shape == (5, 1)
    
    # Test variance
    var = dist.var()
    expected_var = sigma_t.values ** 2
    np.testing.assert_allclose(var.flatten(), expected_var, rtol=1e-10)


def test_temporal_normal_datetime_index():
    """Test TemporalNormal with datetime index."""
    time_index = pd.date_range('2024-01-01', periods=10, freq='D')
    mu_t = pd.Series(np.linspace(0, 10, 10), index=time_index)
    dist = TemporalNormal(mu=mu_t, sigma=1.0)
    
    assert isinstance(dist.index, pd.DatetimeIndex)
    assert len(dist.index) == 10
    assert dist.shape == (10, 1)


def test_temporal_normal_multivariate():
    """Test multivariate TemporalNormal distribution."""
    time_index = pd.RangeIndex(5)
    mu_t = pd.DataFrame([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], index=time_index)
    sigma_t = pd.DataFrame(
        [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2], [1.3, 1.4]],
        index=time_index
    )
    dist = TemporalNormal(mu=mu_t, sigma=sigma_t)
    
    assert dist.shape == (5, 2)
    
    mean = dist.mean()
    assert mean.shape == (5, 2)


def test_temporal_normal_array_input():
    """Test TemporalNormal with array inputs (like Normal)."""
    dist = TemporalNormal(
        mu=[[0, 1], [2, 3], [4, 5]],
        sigma=[[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
    )
    
    assert dist.shape == (3, 2)


def test_temporal_normal_mean_at_time():
    """Test mean_at_time method."""
    mu_t = pd.Series([0, 1, 2, 3, 4], index=pd.RangeIndex(5))
    sigma_t = pd.Series([0.5, 0.7, 1.0, 1.2, 1.5], index=pd.RangeIndex(5))
    dist = TemporalNormal(mu=mu_t, sigma=sigma_t)
    
    # Test mean at specific time points
    mean_0 = dist.mean_at_time(0)
    mean_2 = dist.mean_at_time(2)
    mean_4 = dist.mean_at_time(4)
    
    assert np.isclose(mean_0, 0.0)
    assert np.isclose(mean_2, 2.0)
    assert np.isclose(mean_4, 4.0)


def test_temporal_normal_var_at_time():
    """Test var_at_time method."""
    mu_t = pd.Series([0, 1, 2, 3, 4], index=pd.RangeIndex(5))
    sigma_t = pd.Series([0.5, 0.7, 1.0, 1.2, 1.5], index=pd.RangeIndex(5))
    dist = TemporalNormal(mu=mu_t, sigma=sigma_t)
    
    # Test variance at specific time points
    var_0 = dist.var_at_time(0)
    var_2 = dist.var_at_time(2)
    var_4 = dist.var_at_time(4)
    
    assert np.isclose(var_0, 0.5**2)
    assert np.isclose(var_2, 1.0**2)
    assert np.isclose(var_4, 1.5**2)


def test_temporal_normal_std_at_time():
    """Test std_at_time method."""
    mu_t = pd.Series([0, 1, 2, 3, 4], index=pd.RangeIndex(5))
    sigma_t = pd.Series([0.5, 0.7, 1.0, 1.2, 1.5], index=pd.RangeIndex(5))
    dist = TemporalNormal(mu=mu_t, sigma=sigma_t)
    
    # Test standard deviation at specific time points
    std_0 = dist.std_at_time(0)
    std_2 = dist.std_at_time(2)
    std_4 = dist.std_at_time(4)
    
    assert np.isclose(std_0, 0.5)
    assert np.isclose(std_2, 1.0)
    assert np.isclose(std_4, 1.5)


def test_temporal_normal_datetime_query():
    """Test querying with datetime index."""
    time_index = pd.date_range('2024-01-01', periods=5, freq='D')
    mu_t = pd.Series([0, 1, 2, 3, 4], index=time_index)
    sigma_t = pd.Series([0.5, 0.7, 1.0, 1.2, 1.5], index=time_index)
    dist = TemporalNormal(mu=mu_t, sigma=sigma_t)
    
    # Query by datetime
    query_date = pd.Timestamp('2024-01-03')
    mean_at_date = dist.mean_at_time(query_date)
    var_at_date = dist.var_at_time(query_date)
    
    assert np.isclose(mean_at_date, 2.0)
    assert np.isclose(var_at_date, 1.0**2)


def test_temporal_normal_invalid_time():
    """Test error handling for invalid time queries."""
    mu_t = pd.Series([0, 1, 2, 3, 4], index=pd.RangeIndex(5))
    dist = TemporalNormal(mu=mu_t, sigma=1.0)
    
    # Test out of range index
    with pytest.raises(ValueError):
        dist.mean_at_time(10)
    
    with pytest.raises(ValueError):
        dist.var_at_time(10)


def test_temporal_normal_pdf_cdf():
    """Test pdf and cdf methods work correctly."""
    mu_t = pd.Series([0, 1, 2], index=pd.RangeIndex(3))
    sigma_t = pd.Series([1.0, 1.0, 1.0], index=pd.RangeIndex(3))
    dist = TemporalNormal(mu=mu_t, sigma=sigma_t)
    
    # Test pdf
    x = np.array([[0], [1], [2]])
    pdf = dist.pdf(x)
    assert pdf.shape == (3, 1)
    assert np.all(pdf > 0)
    
    # Test cdf
    cdf = dist.cdf(x)
    assert cdf.shape == (3, 1)
    assert np.all(cdf >= 0)
    assert np.all(cdf <= 1)


def test_temporal_normal_sample():
    """Test sampling from TemporalNormal distribution."""
    mu_t = pd.Series([0, 1, 2, 3, 4], index=pd.RangeIndex(5))
    sigma_t = pd.Series([0.5, 0.7, 1.0, 1.2, 1.5], index=pd.RangeIndex(5))
    dist = TemporalNormal(mu=mu_t, sigma=sigma_t)
    
    # Test single sample
    sample = dist.sample()
    assert sample.shape == (5, 1)
    
    # Test multiple samples
    samples = dist.sample(n_samples=100)
    assert samples.shape[0] == 100
    
    # Check that sample means are roughly correct (with large tolerance)
    sample_means = samples.groupby(level=1).mean()
    # This is a stochastic test, so use loose tolerance
    np.testing.assert_allclose(
        sample_means.values.flatten(), 
        mu_t.values, 
        rtol=0.5, 
        atol=1.0
    )


def test_temporal_normal_scalar():
    """Test TemporalNormal with scalar parameters."""
    dist = TemporalNormal(mu=0.0, sigma=1.0)
    
    assert dist.shape == ()
    assert dist.ndim == 0
    
    # These should work with scalar distribution
    mean = dist.mean()
    var = dist.var()
    
    assert np.isclose(mean, 0.0)
    assert np.isclose(var, 1.0)
