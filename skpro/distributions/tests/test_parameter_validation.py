import pytest
from skpro.distributions.normal import Normal
from skpro.distributions.laplace import Laplace

def test_negative_param_validation():
    """Verify that distributions raise ValueError for non-positive scale parameters."""
    
    # 1. Test Normal distribution boundary condition
    with pytest.raises(ValueError, match="sigma must be strictly positive."):
        Normal(mu=0, sigma=-1)
        
    with pytest.raises(ValueError, match="sigma must be strictly positive."):
        Normal(mu=0, sigma=0)

    # 2. Test Laplace distribution boundary condition
    with pytest.raises(ValueError, match="scale must be strictly positive."):
        Laplace(mu=0, scale=-1)
        
    with pytest.raises(ValueError, match="scale must be strictly positive."):
        Laplace(mu=0, scale=0)