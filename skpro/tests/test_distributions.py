import pandas as pd
import numpy as np
from skpro.distributions.normal import Normal

def test_multiindex_distribution():
    """Test distribution initialization with pd.MultiIndex."""

    # Create hierarchical indices
    index = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["h0", "h1"])
    columns = pd.MultiIndex.from_product([["x", "y"], ["col1", "col2"]], names=["var", "subvar"])

    # Initialize a distribution with MultiIndex
    dist = Normal(index=index, columns=columns, mu=0, sigma=1)

    # 1. Check if indices are preserved
    assert isinstance(dist.index, pd.MultiIndex), "Index is not MultiIndex"
    assert isinstance(dist.columns, pd.MultiIndex), "Columns are not MultiIndex"
    assert dist.index.names == ["h0", "h1"], "Index names mismatch"
    assert dist.columns.names == ["var", "subvar"], "Column names mismatch"

    # 2. Check shape of distribution
    expected_shape = (len(index), len(columns))
    assert dist.shape == expected_shape, f"Expected shape {expected_shape}, got {dist.shape}"

    # 3. Check if `mu` and `sigma` values are correctly initialized
    assert np.all(dist.mu == 0), "Mean (mu) should be 0"
    assert np.all(dist.sigma == 1), "Standard deviation (sigma) should be 1"

    # 4. Test row subsetting (by first-level index)
    subset = dist.loc["a"]
    assert subset.index.equals(pd.Index([1, 2], name="h1")), "Subsetting by 'a' failed"

   