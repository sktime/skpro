"""Generalized Pareto probability distribution for skpro."""

import numpy as np
from scipy.stats import genpareto

from skpro.distributions.base import BaseDistribution


class GeneralizedPareto(BaseDistribution):
    """Generalized Pareto probability distribution.

    Parameters
    ----------
    c : float
        Shape parameter
    scale : float
        Scale parameter
    loc : float
        Location parameter
    """

    _tags = {
        "authors": ["arnavk23"],
        "distr:measuretype": "continuous",
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, c, scale=1.0, loc=0.0, index=None, columns=None):
        self.c = c
        self.scale = scale
        self.loc = loc
        # Ensure public attributes for sklearn compatibility
        self.__dict__["scale"] = scale
        self.__dict__["loc"] = loc
        super().__init__(index=index, columns=columns)

    @property
    def loc(self):
        """Get the location parameter."""
        return self._loc

    @loc.setter
    def loc(self, value):
        """Set the location parameter."""
        self._loc = value

    @property
    def scale(self):
        """Get the scale parameter."""
        return self._scale

    @scale.setter
    def scale(self, value):
        """Set the scale parameter."""
        self._scale = value

    def _pdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        loc = self._bc_params["loc"]
        return genpareto.pdf(x, c, loc=loc, scale=scale)

    def _cdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        loc = self._bc_params["loc"]
        return genpareto.cdf(x, c, loc=loc, scale=scale)

    def _ppf(self, p):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        loc = self._bc_params["loc"]
        return genpareto.ppf(p, c, loc=loc, scale=scale)

    def _mean(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        loc = self._bc_params["loc"]
        return genpareto.mean(c, loc=loc, scale=scale)

    def _var(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        loc = self._bc_params["loc"]
        import numpy as np

        v = genpareto.var(c, loc=loc, scale=scale)
        return v if np.isfinite(v) and v >= 0 else np.inf

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For GeneralizedPareto(c, scale, loc), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \int_{lb}^{ub} F(t)(1-F(t))\,dt

        using numerical integration. The upper bound is inf for c>=0 and
        loc + scale/|c| for c<0.
        """
        from scipy.integrate import quad

        c = np.asarray(self._bc_params["c"])
        scale = np.asarray(self._bc_params["scale"])
        loc = np.asarray(self._bc_params["loc"])
        c_b, scale_b, loc_b = np.broadcast_arrays(c, scale, loc)
        result = np.empty_like(c_b, dtype=float)

        it = np.nditer(
            [c_b, scale_b, loc_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
        )
        for cc, ss, ll, out in it:
            cc_val = float(cc)
            ss_val = float(ss)
            ll_val = float(ll)
            lb = ll_val
            ub = np.inf if cc_val >= 0 else ll_val + ss_val / abs(cc_val)

            def integrand(t, cc=cc_val, ss=ss_val, ll=ll_val):
                F = genpareto.cdf(t, cc, loc=ll, scale=ss)
                return 2 * F * (1 - F)

            val, _ = quad(integrand, lb, ub, limit=200)
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

        :math:`\mathbb{E}[|X-x|]` for X ~ GeneralizedPareto(c, scale, loc),
        computed via numerical integration.
        """
        from scipy.integrate import quad

        c = np.asarray(self._bc_params["c"])
        scale = np.asarray(self._bc_params["scale"])
        loc = np.asarray(self._bc_params["loc"])
        x_arr = np.asarray(x)
        c_b, scale_b, loc_b = np.broadcast_arrays(c, scale, loc)
        _, x_b = np.broadcast_arrays(c_b, x_arr)
        result = np.empty_like(c_b, dtype=float)

        it = np.nditer(
            [c_b, scale_b, loc_b, x_b, result],
            flags=["multi_index"],
            op_flags=[
                ["readonly"],
                ["readonly"],
                ["readonly"],
                ["readonly"],
                ["writeonly"],
            ],
        )
        for cc, ss, ll, x0, out in it:
            cc_val = float(cc)
            ss_val = float(ss)
            ll_val = float(ll)
            x0_val = float(x0)
            lb = ll_val
            ub = np.inf if cc_val >= 0 else ll_val + ss_val / abs(cc_val)

            def integrand(t, cc=cc_val, ss=ss_val, ll=ll_val, x0=x0_val):
                return abs(t - x0) * genpareto.pdf(t, cc, loc=ll, scale=ss)

            val, _ = quad(integrand, lb, ub, limit=200)
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
        """Return test parameters for GeneralizedPareto."""
        params1 = {"c": 0.5, "scale": 1.0, "loc": 0.0}
        params2 = {"c": 1.0, "scale": 2.0, "loc": 1.0}
        return [params1, params2]
