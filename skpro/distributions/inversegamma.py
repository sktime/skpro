# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Inverse Gamma probability distribution."""

__author__ = ["meraldoantonio"]

import pandas as pd
from scipy.stats import invgamma, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class InverseGamma(_ScipyAdapter):
    r"""Inverse Gamma Distribution.

    Most methods wrap ``scipy.stats.invgamma``.

    The Inverse Gamma Distribution is parameterized by shape :math:`\alpha` and
    scale :math:`\beta`, such that the pdf is

    .. math:: f(x) = \frac{\beta^{\alpha} x^{-\alpha-1} \exp\left(-\frac{\beta}{x}\right)}{\tau(\alpha)}

    where :math:`\tau(\alpha)` is the Gamma function.
    For all positive integers, :math:`\tau(\alpha) = (\alpha-1)!`.

    Parameters
    ----------
    alpha : float or array of float (1D or 2D)
        The shape parameter.
    beta : float or array of float (1D or 2D)
        The scale parameter.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.inversegamma import InverseGamma

    >>> d = InverseGamma(beta=[[1, 1], [2, 3], [4, 5]], alpha=2)
    """  # noqa: E501

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, alpha, beta, index=None, columns=None):
        self.alpha = alpha
        self.beta = beta
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return invgamma

    def _get_scipy_param(self):
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]
        scale = beta
        return [], {"a": alpha, "scale": scale}

    def _energy_self(self):
        """Energy of self, w.r.t. self (expected |X-Y| for i.i.d. X,Y ~ InverseGamma)."""
        import numpy as np
        from scipy.integrate import quad
        alpha = np.asarray(self._bc_params["alpha"])
        beta = np.asarray(self._bc_params["beta"])
        # Broadcast alpha and beta to the same shape
        alpha_b, beta_b = np.broadcast_arrays(alpha, beta)
        result = np.empty_like(alpha_b, dtype=float)
        it = np.nditer([alpha_b, beta_b, result], flags=["multi_index"], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
        for a, b, out in it:
            def cdf(x):
                from scipy.stats import invgamma
                return invgamma.cdf(x, a=a.item(), scale=b.item())
            def integrand(x):
                F = cdf(x)
                return 2 * F * (1 - F)
            val, _ = quad(integrand, 0, np.inf, limit=200)
            out[...] = val
        # Always flatten to 1D of length n_rows for DataFrame compatibility
        n_rows = 1 if self.index is None else len(self.index)
        result = np.asarray(result).reshape(-1)
        if result.shape[0] != n_rows:
            result = result.reshape(n_rows, -1).mean(axis=1)
        if self.index is None and n_rows == 1:
            return result.item()
        return result

    def _energy_x(self, x):
        """Energy of self, w.r.t. a constant frame x (expected |X-x| for X ~ InverseGamma)."""
        import numpy as np
        from scipy.stats import invgamma
        from scipy.integrate import quad
        alpha = np.asarray(self._bc_params["alpha"])
        beta = np.asarray(self._bc_params["beta"])
        x = np.asarray(x)
        # Broadcast all to the same shape
        alpha_b, beta_b, x_b = np.broadcast_arrays(alpha, beta, x)
        result = np.empty_like(alpha_b, dtype=float)
        it = np.nditer([alpha_b, beta_b, x_b, result], flags=["multi_index"], op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]])
        for a, b, x0, out in it:
            def integrand(t):
                return np.abs(t - x0.item()) * invgamma.pdf(t, a=a.item(), scale=b.item())
            val, _ = quad(integrand, 0, np.inf, limit=200)
            out[...] = val
        # Always flatten to 1D of length n_rows for DataFrame compatibility
        n_rows = 1 if self.index is None else len(self.index)
        result = np.asarray(result).reshape(-1)
        if result.shape[0] != n_rows:
            result = result.reshape(n_rows, -1).mean(axis=1)
        if self.index is None and n_rows == 1:
            return result.item()
        return result

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }


    def __init__(self, alpha, beta, index=None, columns=None):
        self.alpha = alpha
        self.beta = beta
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return invgamma

    def _get_scipy_param(self):
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]
        scale = beta

        return [], {"a": alpha, "scale": scale}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"alpha": [6, 2.5], "beta": [[1, 1], [2, 3], [4, 5]]}
        params2 = {
            "alpha": 2,
            "beta": 3,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"alpha": 1.5, "beta": 2.1}

        return [params1, params2, params3]
