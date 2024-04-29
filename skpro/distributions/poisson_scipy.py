from scipy.stats import rv_discrete, poisson
from skpro.distributions.adapters.scipy import _ScipyAdapter

__all__ = ["PoissonScipy"]

class PoissonScipy(_ScipyAdapter):
    _tags = {
        "capabilities:approx": ["ppf", "energy"],
        "capabilities:exact": ["mean", "var", "pmf", "log_pmf", "cdf"],
        "distr:measuretype": "discrete",
        "broadcast_init": "on",
    }

    def __init__(self, mu, index=None, columns=None):
        self.mu = mu

        super().__init__(index=index, columns=columns)
    
    def _get_scipy_object(self) -> rv_discrete:
        return poisson

    def _get_scipy_param(self) -> dict:
        mu = self._bc_params["mu"]

        return [mu], {}
