# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Inverse Gaussian probability distribution."""

__author__ = ["Omswastik-11"]

import pandas as pd
from scipy.stats import invgauss, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class InverseGaussian(_ScipyAdapter):
    r"""Inverse Gaussian distribution, aka Wald distribution.

    Most methods wrap ``scipy.stats.invgauss``.

    The Inverse Gaussian distribution (Wald) when using SciPy's
    parameterization is specified by a shape parameter ``mu`` and a
    ``scale`` parameter. In SciPy these are the positional and keyword
    parameters of ``scipy.stats.invgauss(mu, scale=scale)``. The
    mean of the distribution is given by ``mean = mu * scale``.

    The pdf in terms of :math:`\mu` = ``mu`` and :math:`\sigma` = ``scale`` is:

    .. math:: f(x; \mu, \sigma) = \sqrt{\frac{\sigma}{2 \pi x^3}}
              \exp\left(-\frac{(x - \mu \sigma)^2}{2 \mu^2 \sigma x}\right)

    Parameters
    ----------
    mu : float or array of float (1D or 2D), must be positive
        shape parameter (dimensionless)
    scale : float or array of float (1D or 2D), must be positive
        scale parameter (multiplies the distribution)
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.inversegaussian import InverseGaussian

    >>> d = InverseGaussian(mu=1.0, scale=1.0)
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, scale, index=None, columns=None):
        self.mu = mu
        self.scale = scale

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return invgauss

    def _get_scipy_param(self):
        # Pass parameters directly to scipy.stats.invgauss.
        # SciPy's invgauss accepts a shape parameter `mu` and a keyword  `scale`.
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]

        return [mu], {"scale": scale}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [2, 3.5], "scale": [[1, 1], [2, 3], [4, 5]]}
        params2 = {
            "mu": 2.5,
            "scale": 1.5,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": 3.0, "scale": 2.0}

        return [params1, params2, params3]
