from scipy.stats import rv_discrete, poisson
from skpro.distributions.adapters.scipy import _ScipyDiscreteAdapter

__all__ = ["PoissonScipy"]

class PoissonScipy(_ScipyDiscreteAdapter):
    """Poisson distribution.

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        mean of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions import PoissonScipy as Poisson

    >>> distr = Poisson(mu=[[1, 1], [2, 3], [4, 5]])
    """

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
