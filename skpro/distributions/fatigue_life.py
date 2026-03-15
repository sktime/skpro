"""Fatigue-life (Birnbaum–Saunders) probability distribution for skpro."""

import numpy as np
from scipy.stats import fatiguelife, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class FatigueLife(_ScipyAdapter):
    r"""Fatigue-life (Birnbaum–Saunders) probability distribution.

    The fatigue-life (Birnbaum–Saunders) distribution is a continuous probability
    distribution used to model failure times under cyclic stress. It is parameterized
    by a shape parameter $c > 0$ and a scale parameter $s > 0$. Its probability
    density function (PDF) is:

    .. math::
        f(x; c, s) = \frac{1}{2 c s \sqrt{2 \pi}}
        \left( \frac{\sqrt{x/s} + \sqrt{s/x}}{x} \right)
        \exp\left( -\frac{(\sqrt{x/s} - \sqrt{s/x})^2}{2 c^2} \right),
        \quad x > 0

    Parameters
    ----------
    c : float
        Shape parameter
    scale : float
        Scale parameter
    """

    _tags = {
        "authors": ["arnavk23"],
        "distr:measuretype": "continuous",
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, c, scale=1.0, index=None, columns=None):
        self.c = c
        self.scale = scale
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return fatiguelife

    def _get_scipy_param(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return [], {"c": c, "scale": scale}

    def _pdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return fatiguelife.pdf(x, c, scale=scale)

    def _cdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return fatiguelife.cdf(x, c, scale=scale)

    def _ppf(self, p):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return fatiguelife.ppf(p, c, scale=scale)

    def _mean(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return fatiguelife.mean(c, scale=scale)

    def _var(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return fatiguelife.var(c, scale=scale)

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For FatigueLife(c, scale), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t))\,dt

        using numerical integration.
        """
        from scipy.integrate import quad

        c = np.asarray(self._bc_params["c"])
        scale = np.asarray(self._bc_params["scale"])
        c_b, scale_b = np.broadcast_arrays(c, scale)
        result = np.empty_like(c_b, dtype=float)

        it = np.nditer(
            [c_b, scale_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for cc, ss, out in it:
            cc_val = float(cc)
            ss_val = float(ss)

            def integrand(t, cc=cc_val, ss=ss_val):
                F = fatiguelife.cdf(t, c=cc, scale=ss)
                return 2 * F * (1 - F)

            val, _ = quad(integrand, 0, np.inf, limit=200)
            out[...] = val

        result_flat = np.asarray(result).reshape(-1)
        n_rows = 1 if self.index is None else len(self.index)
        if result_flat.shape[0] != n_rows:
            result_flat = result_flat.reshape(n_rows, -1).sum(axis=1)
        if self.index is None and n_rows == 1:
            return float(result_flat[0])
        return result_flat

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|]` for X ~ FatigueLife(c, scale),
        computed via numerical integration.
        """
        from scipy.integrate import quad

        c = np.asarray(self._bc_params["c"])
        scale = np.asarray(self._bc_params["scale"])
        x_arr = np.asarray(x)
        c_b, scale_b = np.broadcast_arrays(c, scale)
        _, x_b = np.broadcast_arrays(c_b, x_arr)
        result = np.empty_like(c_b, dtype=float)

        it = np.nditer(
            [c_b, scale_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
        )
        for cc, ss, x0, out in it:
            cc_val = float(cc)
            ss_val = float(ss)
            x0_val = float(x0)

            def integrand(t, cc=cc_val, ss=ss_val, x0=x0_val):
                return abs(t - x0) * fatiguelife.pdf(t, c=cc, scale=ss)

            val, _ = quad(integrand, 0, np.inf, limit=200)
            out[...] = val

        result_flat = np.asarray(result).reshape(-1)
        n_rows = 1 if self.index is None else len(self.index)
        if result_flat.shape[0] != n_rows:
            result_flat = result_flat.reshape(n_rows, -1).sum(axis=1)
        if self.index is None and n_rows == 1:
            return float(result_flat[0])
        return result_flat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for FatigueLife."""
        params1 = {"c": 2.0, "scale": 1.0}
        params2 = {"c": 1.5, "scale": 2.0}
        return [params1, params2]
