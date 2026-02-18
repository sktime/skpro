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
    
    reg = HistogramCDERegressor(n_bins_x=2, n_bins_y=5)
    reg.fit(X, y)
    
    X_test = pd.DataFrame({"a": [1.5], "b": [4.5]})
    y_dist = reg.predict_proba(X_test)
    
    assert isinstance(y_dist, Histogram)
    assert y_dist.shape == (1, 1)
    
    # For Histogram distribution, bin_mass is exposed as nested list
    # Check that it sums to 1 for the first instance and column
    assert np.allclose(y_dist.bin_mass[0][0].sum(), 1.0)
