# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Inverse Gaussian probability distribution."""

__author__ = ["Omswastik-11"]

import pandas as pd
from scipy.stats import invgauss, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class InverseGaussian(_ScipyAdapter):
    r"""Inverse Gaussian distribution, aka Wald distribution.

    Most methods wrap ``scipy.stats.invgauss``.

    The Inverse Gaussian distribution is parameterized by mean :math:`\mu` and
    shape parameter :math:`\lambda`, such that the pdf is

    .. math:: f(x) = \sqrt{\frac{\lambda}{2\pi x^3}}
              \exp\left(-\frac{\lambda(x-\mu)^2}{2\mu^2 x}\right)

    The mean :math:`\mu` is represented by the parameter ``mu``,
    and the shape parameter :math:`\lambda` by the parameter ``lambda_``.

    Note: This parameterization corresponds to
    ``numpy.random.wald(mean=mu, scale=lambda_)``.
    In ``scipy.stats.invgauss(mu_scipy, scale=scale_scipy)``, the parameters are
    related as:
    ``scale_scipy = lambda_``
    ``mu_scipy = mu / lambda_``

    Parameters
    ----------
    mu : float or array of float (1D or 2D), must be positive
        mean of the distribution
    lambda_ : float or array of float (1D or 2D), must be positive
        shape parameter (lambda) of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.inversegaussian import InverseGaussian

    >>> d = InverseGaussian(mu=[[1, 1], [2, 3], [4, 5]], lambda_=1)
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, lambda_, index=None, columns=None):
        self.mu = mu
        self.lambda_ = lambda_

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return invgauss

    def _get_scipy_param(self):
        mu = self._bc_params["mu"]
        lambda_ = self._bc_params["lambda_"]

        # Mapping to scipy parameters
        # scipy_scale = lambda_
        # scipy_mu = mu / lambda_

        scipy_scale = lambda_
        scipy_mu = mu / lambda_

        return [scipy_mu], {"scale": scipy_scale}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [2, 3.5], "lambda_": [[1, 1], [2, 3], [4, 5]]}
        params2 = {
            "mu": 2.5,
            "lambda_": 1.5,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": 3.0, "lambda_": 2.0}

        return [params1, params2, params3]
