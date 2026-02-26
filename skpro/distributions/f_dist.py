"""F probability distribution for skpro."""

from scipy.stats import f, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class FDist(_ScipyAdapter):
    r"""F (Fisher-Snedecor) probability distribution.

    The F-distribution (Fisher-Snedecor distribution) is a continuous probability
    distribution that arises frequently as the null distribution of a test statistic,
    especially in ANOVA. It is parameterized by two degrees of freedom parameters
    $d_1 > 0$ (numerator) and $d_2 > 0$ (denominator). Its probability density
    function (PDF) is:

    .. math::
        f(x; d_1, d_2) =
        \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1 + d_2}}}}
        {x \cdot B\left(\frac{d_1}{2}, \frac{d_2}{2}\right)}, \quad x > 0

    where $B$ is the beta function.

    Parameters
    ----------
    dfn : float
        Degrees of freedom numerator
    dfd : float
        Degrees of freedom denominator
    """

    _tags = {
        "authors": ["arnavk23"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, dfn, dfd, index=None, columns=None):
        self.dfn = dfn
        self.dfd = dfd
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return f

    def _get_scipy_param(self):
        dfn = self._bc_params["dfn"]
        dfd = self._bc_params["dfd"]
        return [], {"dfn": dfn, "dfd": dfd}

    def _pdf(self, x):
        dfn = self._bc_params["dfn"]
        dfd = self._bc_params["dfd"]
        return f.pdf(x, dfn, dfd)

    def _cdf(self, x):
        dfn = self._bc_params["dfn"]
        dfd = self._bc_params["dfd"]
        return f.cdf(x, dfn, dfd)

    def _ppf(self, p):
        dfn = self._bc_params["dfn"]
        dfd = self._bc_params["dfd"]
        return f.ppf(p, dfn, dfd)

    def _mean(self):
        dfn = self._bc_params["dfn"]
        dfd = self._bc_params["dfd"]
        return f.mean(dfn, dfd)

    def _var(self):
        dfn = self._bc_params["dfn"]
        dfd = self._bc_params["dfd"]
        return f.var(dfn, dfd)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for FDist."""
        params1 = {"dfn": 5.0, "dfd": 2.0}
        params2 = {"dfn": 10.0, "dfd": 5.0}
        return [params1, params2]
