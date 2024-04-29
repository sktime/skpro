from scipy.stats import rv_continuous, fisk
from skpro.distributions.adapters.scipy import _ScipyAdapter

__all__ = ["FiskScipy"]

class FiskScipy(_ScipyAdapter):
    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
    }

    def __init__(self, alpha=1, beta=1, index=None, columns=None):
        self.alpha = alpha
        self.beta = beta
    
        super().__init__(index=index, columns=columns)
    
    def _get_scipy_object(self) -> rv_continuous:
        return fisk
    
    def _get_scipy_param(self) -> dict:
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        return {"c": beta, "scale": alpha}
