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
    - k : int or array of int (1D or 2D)  
    Represents the shape parameter.
    - lambda_ : float or array of float (1D or 2D)  
    Represents the rate parameter, which is also the inverse of the scale parameter.
    - index : pd.Index, optional, default = RangeIndex
    - columns : pd.Index, optional, default = RangeIndex  

    Examples
    ----------
    >>> from skpro.distributions.erlang import Erlang

    >>> d = Erlang(lambda_=[[1, 1], [2, 3], [4, 5]], k=2)
    """
    
    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }
    def __init__(self,lambda_, k, index=None, columns=None):
        if(lambda_ <= 0):
            raise ValueError("lambda_ must be greater than 0.")
        if( isinstance(k, int) == False or k <= 0):
            raise ValueError("k must be a positive integer.")
        self.lambda_ = lambda_
        self.k = k
        
        super().__init__(index=index, columns=columns)
    def _get_scipy_object(self) -> rv_continuous:
        return erlang
    
    def _get_scipy_param(self):
        lambda_ = self._bc_params["lambda_"]
        k = self._bc_params["k"]
        return [],{"scale":1/lambda_,"a":k}
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # Array case examples
        params1 = {
            "lambda_": 2.0,
            "k": 3,          
            "index": pd.Index([0, 1, 2]),  
            "columns": pd.Index(["x", "y"]),  
        }
        # Scalar case examples
        params2 = {
            "lambda_": 0.8,
            "k": 2
        }
        
        params3 = {
            "lambda_": 3.0,
            "k": 1
        }

        return [params1, params2, params3]
