"""Prior specification for Bayesian model parameters.

Wraps skpro distribution objects with parameter metadata,
providing a unified prior specification language across all
Bayesian estimators.
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["david_laid"]


class Prior:
    """Prior specification for a single model parameter.

    Wraps a skpro ``BaseDistribution`` instance with a parameter name,
    allowing Bayesian estimators to accept priors in a backend-agnostic way.

    Parameters
    ----------
    distribution : BaseDistribution
        A skpro distribution instance (e.g., ``Normal(mu=0, sigma=10)``).
    name : str, optional
        Name of the model parameter this prior applies to.

    Examples
    --------
    >>> from skpro.distributions import Normal
    >>> from skpro.regression.bayesian._prior import Prior
    >>> prior = Prior(Normal(mu=0, sigma=10), name="coefficients")
    >>> prior.sample(3).shape  # doctest: +SKIP
    (3, 1, 1)
    """

    def __init__(self, distribution, name=None):
        from skpro.distributions.base import BaseDistribution

        if not isinstance(distribution, BaseDistribution):
            raise TypeError(
                f"`distribution` must be a skpro BaseDistribution, "
                f"got {type(distribution).__name__}"
            )
        self.distribution = distribution
        self.name = name

    def sample(self, n_samples=1):
        """Draw samples from the prior distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.

        Returns
        -------
        samples : pd.DataFrame
            Samples from the prior distribution.
        """
        return self.distribution.sample(n_samples)

    def log_pdf(self, x):
        """Evaluate the log-density of the prior at ``x``.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the log-density.

        Returns
        -------
        log_density : pd.DataFrame
            Log-density values.
        """
        return self.distribution.log_pdf(x)

    def mean(self):
        """Return the prior mean."""
        return self.distribution.mean()

    def var(self):
        """Return the prior variance."""
        return self.distribution.var()

    def __repr__(self):
        dist_name = type(self.distribution).__name__
        name_str = f", name='{self.name}'" if self.name is not None else ""
        return f"Prior({dist_name}{name_str})"
