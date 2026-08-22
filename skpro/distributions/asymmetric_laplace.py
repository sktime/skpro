# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Asymmetric Laplace probability distribution."""

__author__ = ["Ashish-Kumar-Dash"]
__all__ = ["AsymmetricLaplace"]

import numpy as np
import pandas as pd
from scipy.stats import laplace_asymmetric, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class AsymmetricLaplace(_ScipyAdapter):
    r"""Asymmetric Laplace distribution.

    Most methods wrap ``scipy.stats.laplace_asymmetric``.

    The Asymmetric Laplace distribution is parametrized by asymmetry
    :math:`\kappa > 0`, location :math:`\mu`, and scale :math:`\sigma > 0`,
    such that the pdf is

    .. math::

        f(x; \kappa, \mu, \sigma) = \frac{1}{\sigma(\kappa + \kappa^{-1})}
        \begin{cases}
            \exp\left(-\kappa \frac{x - \mu}{\sigma}\right),
                & x \geq \mu \\
            \exp\left(\frac{x - \mu}{\kappa \sigma}\right),
                & x < \mu
        \end{cases}

    For :math:`\kappa = 1`, this reduces to the symmetric Laplace distribution.

    Parameters
    ----------
    kappa : float or array of float (1D or 2D), must be positive
        asymmetry parameter; ``kappa = 1`` gives the symmetric Laplace
    mu : float or array of float (1D or 2D)
        location parameter of the distribution
    scale : float or array of float (1D or 2D), must be positive
        scale parameter of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions import AsymmetricLaplace

    >>> d = AsymmetricLaplace(kappa=2, mu=0, scale=1)
    """

    _tags = {
        "authors": ["Ashish-Kumar-Dash"],
        "capabilities:exact": [
            "mean",
            "var",
            "energy",
            "pdf",
            "log_pdf",
            "cdf",
            "ppf",
        ],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
    }

    def __init__(self, kappa, mu=0, scale=1, index=None, columns=None):
        self.kappa = kappa
        self.mu = mu
        self.scale = scale

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return laplace_asymmetric

    def _get_scipy_param(self):
        kappa = self._bc_params["kappa"]
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]
        return [kappa], {"loc": mu, "scale": scale}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        K = self._bc_params["kappa"]
        sc = self._bc_params["scale"]

        energy_arr = sc * (K**4 + K**2 + 1) / (K * (1 + K**2))
        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|]`, where :math:`X` is a copy of self,
        and :math:`x` is a constant.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to compute energy w.r.t. to

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        K = self._bc_params["kappa"]
        mu = self._bc_params["mu"]
        sc = self._bc_params["scale"]

        z = (x - mu) / sc
        pos = z >= 0

        energy_arr = np.where(
            pos,
            z
            + (K**2 - 1) / K
            + 2 * np.exp(-K * np.where(pos, z, 0)) / (K * (1 + K**2)),
            -z
            - (K**2 - 1) / K
            + 2 * K**3 * np.exp(np.where(pos, 0, z) / K) / (1 + K**2),
        )
        energy_arr = sc * energy_arr

        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"kappa": [[0.5, 2], [1, 3], [2, 0.5]], "mu": 0, "scale": 1}
        params2 = {
            "kappa": 2,
            "mu": 0,
            "scale": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        params3 = {"kappa": 0.5, "mu": -1, "scale": 2}

        return [params1, params2, params3]
