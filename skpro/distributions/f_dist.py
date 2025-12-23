"""F probability distribution for skpro."""

from scipy.stats import f

from skpro.distributions.base import BaseDistribution


class FDist(BaseDistribution):
    """F (Fisher-Snedecor) probability distribution.

    Parameters
    ----------
    dfn : float
        Degrees of freedom numerator
    dfd : float
        Degrees of freedom denominator
    """

    _tags = {
        "authors": ["your-github-id"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, dfn, dfd, index=None, columns=None):
        self.dfn = dfn
        self.dfd = dfd
        super().__init__(index=index, columns=columns)

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
