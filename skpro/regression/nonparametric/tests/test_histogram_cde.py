# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
import pytest
import pandas as pd
import numpy as np
from skpro.regression.nonparametric import HistogramCDERegressor
from skpro.distributions import Histogram

def test_histogram_cde_output():
    """Test that HistogramCDERegressor returns a Histogram distribution with correct shape."""
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = pd.Series([10, 20, 30], name="y").to_frame()
    
    reg = HistogramCDERegressor(n_neighbors=2, n_bins_y=5)
    reg.fit(X, y)
    
    X_test = pd.DataFrame({"a": [1.5], "b": [4.5]})
    y_dist = reg.predict_proba(X_test)
    
    assert isinstance(y_dist, Histogram)
    assert y_dist.shape == (1, 1)
    
    # For Histogram distribution, bin_mass is exposed as nested list
    # Check that it sums to 1 for the first instance and column
    assert np.allclose(y_dist.bin_mass[0][0].sum(), 1.0)


def test_histogram_cde_constant_target():
    """Test with constant target to ensure degenerate bins are handled."""
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([10, 10, 10], name="y").to_frame()
    
    reg = HistogramCDERegressor(n_neighbors=2, n_bins_y=5)
    reg.fit(X, y)
    
    X_test = pd.DataFrame({"a": [2]})
    y_dist = reg.predict_proba(X_test)
    
    assert y_dist.shape == (1, 1)
    assert np.allclose(y_dist.bin_mass[0][0].sum(), 1.0)


def test_histogram_cde_multiple_query():
    """Test with multiple query points."""
    np.random.seed(42)
    X = pd.DataFrame({"a": np.random.randn(20)})
    y = pd.Series(np.random.randn(20), name="y").to_frame()
    
    reg = HistogramCDERegressor(n_neighbors=5, n_bins_y=5)
    reg.fit(X, y)
    
    X_test = pd.DataFrame({"a": np.random.randn(5)})
    y_dist = reg.predict_proba(X_test)
    
    assert y_dist.shape == (5, 1)
    for i in range(5):
        assert np.allclose(y_dist.bin_mass[i][0].sum(), 1.0)

