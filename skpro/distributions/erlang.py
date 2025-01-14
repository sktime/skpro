# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Erlang probability distribution."""

__author__ = ["RUPESH-KUMAR01"]

import pandas as pd
from scipy.stats import erlang, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter

class Erlang(_ScipyAdapter):
    r"""Erlang Distribution

    Most methods wrap ``scipy.stats.erlang``.

    The Erlang Distribution is parameterized by shape :math:`k` 
    and rate :math:`\lambda`, such that the pdf is

    .. math:: f(x) = \frac{x^{k-1}\exp\left(-\lambda x\right) \lambda^{k}}{(k-1)!}


    Parameters
    ----------
    - shape : int or array of int (1D or 2D)  
        Represents the shape parameter.
    - rate : float or array of float (1D or 2D)  
        Represents the rate parameter, which is also the inverse of the scale parameter.
    - index : pd.Index, optional, default = RangeIndex
    - columns : pd.Index, optional, default = RangeIndex  

    Examples
    ----------
    >>> from skpro.distributions.erlang import Erlang

    >>> d = Erlang(rate=[[1, 1], [2, 3], [4, 5]], shape=2)
    """
    
    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }
    def __init__(self,rate, shape, index=None, columns=None):
        if(rate <= 0):
            raise ValueError("Rate must be greater than 0.")
        if( isinstance(shape, int) == False or shape <= 0):
            raise ValueError("shape must be a positive integer.")
        self.rate = rate
        self.shape = shape
        
        super().__init__(index=index, columns=columns)
    def _get_scipy_object(self) -> rv_continuous:
        return erlang
    
    def _get_scipy_param(self):
        rate = self._bc_params["rate"]
        shape = self._bc_params["shape"]
        return [],{"scale":1/rate,"a":shape}
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # Array case examples
        params1 = {
            "rate": 2.0,
            "shape": 3,          
            "index": pd.Index([0, 1, 2]),  
            "columns": pd.Index(["x", "y"]),  
        }
        # Scalar case examples
        params2 = {
            "rate": 0.8,
            "shape": 2
        }
        
        params3 = {
            "rate": 3.0,
            "shape": 1
        }

        return [params1, params2, params3]
