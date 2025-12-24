# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Log-Gamma probability distribution."""

__author__ = ["ali-john"]

import pandas as pd
from scipy.stats import loggamma, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class LogGamma(_ScipyAdapter):
    r"""Log-Gamma Distribution.

    Most methods wrap ``scipy.stats.loggamma``.

    The Log-Gamma distribution is a continuous probability distribution
    whose logarithm is related to the gamma distribution. It is useful
    in extreme value theory and reliability analysis.

    The Log-Gamma distribution is parameterized by the shape parameter
    :math:`c`, such that the pdf is

    .. math:: f(x) = \frac{\exp(cx - \exp(x))}{\Gamma(c)}

    where :math:`\Gamma(c)` is the Gamma function.

    The shape parameter :math:`c` is represented by the parameter ``c``.

    Parameters
    ----------
    c : float or array of float (1D or 2D), must be positive
        shape parameter of the log-gamma distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.loggamma import LogGamma

    >>> d = LogGamma(c=[[1, 2], [3, 4], [5, 6]])
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, c, index=None, columns=None):
        self.c = c

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return loggamma

    def _get_scipy_param(self):
        c = self._bc_params["c"]

        return [c], {}

    def _energy_self(self):
        """Energy of self, w.r.t. self (expected |X-Y| for i.i.d. X,Y ~ LogGamma)."""
        import numpy as np
        from scipy.integrate import quad
        from scipy.stats import loggamma

        c = np.asarray(self._bc_params["c"])
        c_b = np.broadcast_to(c, c.shape)
        result = np.empty_like(c_b, dtype=float)
        it = np.nditer(
            [c_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["writeonly"]],
        )
        for cc, out in it:
            cc_val = cc.item()

            def cdf(x, cc_val=cc_val):
                return loggamma.cdf(x, cc_val)

            def integrand(x, cdf=cdf):
                F = cdf(x)
                return 2 * F * (1 - F)

            val, _ = quad(integrand, -np.inf, np.inf, limit=200)
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
        """Energy of self, w.r.t. a constant frame x.

        Expected |X-x| for X ~ LogGamma.
        """
        import numpy as np
        from scipy.integrate import quad
        from scipy.stats import loggamma

        c = np.asarray(self._bc_params["c"])
        x = np.asarray(x)
        c_b, x_b = np.broadcast_arrays(c, x)
        result = np.empty_like(c_b, dtype=float)
        it = np.nditer(
            [c_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for cc, x0, out in it:
            cc_val = cc.item()
            x0_val = x0.item()

            def integrand(t, cc_val=cc_val, x0_val=x0_val):
                return np.abs(t - x0_val) * loggamma.pdf(t, cc_val)

            val, _ = quad(integrand, -np.inf, np.inf, limit=200)
            out[...] = val
        # Always flatten to 1D of length n_rows for DataFrame compatibility
        n_rows = 1 if self.index is None else len(self.index)
        result = np.asarray(result).reshape(-1)
        if result.shape[0] != n_rows:
            result = result.reshape(n_rows, -1).mean(axis=1)
        if self.index is None and n_rows == 1:
            return result.item()
        return result

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"c": [[1, 2], [3, 4]]}
params2 = {
            "c": 2,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"c": 1.5}

        return [params1, params2, params3]
