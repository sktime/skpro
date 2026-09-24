"""Base class for Bayesian probabilistic regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["david_laid"]

import pandas as pd

from skpro.regression.base import BaseProbaRegressor


class BaseBayesianRegressor(BaseProbaRegressor):
    """Base mixin for Bayesian probabilistic regressors.

    Extends ``BaseProbaRegressor`` with standardized interfaces for:

    * Prior specification and access (``get_prior``, ``set_prior``)
    * Posterior access (``get_posterior``, ``get_posterior_summary``)
    * Posterior sampling (``sample_posterior``)
    * Sequential Bayesian updating (inherits ``update`` from base class)

    Subclasses must implement
    -------------------------
    _fit(X, y, C=None)
        Perform posterior inference given training data.
    _predict_proba(X)
        Return the posterior predictive distribution.
    _get_prior_params()
        Return a ``dict[str, BaseDistribution]`` of prior distributions.
    _get_posterior_params()
        Return a ``dict[str, BaseDistribution]`` of posterior distributions.

    Optionally override
    -------------------
    _sample_posterior(n_samples)
        Draw from the parameter posterior (default uses distribution sampling).
    _update(X, y, C=None)
        Efficient sequential Bayesian update.
    """

    _tags = {
        "capability:update": True,
    }

    # --- Prior interface ---------------------------------------------------

    def get_prior(self):
        """Return prior distributions over model parameters.

        Returns
        -------
        prior : dict of str -> BaseDistribution
            Mapping from parameter name to its prior distribution.
        """
        return self._get_prior_params()

    def _get_prior_params(self):
        """Return prior distributions. Override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_prior_params"
        )

    def set_prior(self, **priors):
        """Set prior distributions for model parameters.

        Parameters
        ----------
        **priors : BaseDistribution or Prior
            Prior distributions keyed by parameter name.

        Returns
        -------
        self
        """
        if not hasattr(self, "_custom_priors"):
            self._custom_priors = {}
        self._custom_priors.update(priors)
        return self

    # --- Posterior interface ------------------------------------------------

    def get_posterior(self):
        """Return posterior distributions over model parameters.

        Must be called after ``fit``.

        Returns
        -------
        posterior : dict of str -> BaseDistribution
            Mapping from parameter name to its posterior distribution.
        """
        self.check_is_fitted()
        return self._get_posterior_params()

    def _get_posterior_params(self):
        """Return posterior distributions. Override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_posterior_params"
        )

    def get_posterior_summary(self):
        """Return summary statistics of the posterior.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with mean, std, and 95% credible interval bounds
            for each model parameter.
        """
        self.check_is_fitted()
        return self._get_posterior_summary()

    def _get_posterior_summary(self):
        """Compute posterior summary. Override for custom behaviour."""
        posterior = self._get_posterior_params()
        rows = []
        for name, dist in posterior.items():
            mean_val = dist.mean()
            var_val = dist.var()
            q_lo = dist.ppf(0.025)
            q_hi = dist.ppf(0.975)

            # Flatten to scalar when possible
            def _scalar(v):
                try:
                    return float(v.values.ravel()[0])
                except (AttributeError, IndexError):
                    return float(v)

            rows.append(
                {
                    "parameter": name,
                    "mean": _scalar(mean_val),
                    "std": _scalar(var_val) ** 0.5,
                    "q_0.025": _scalar(q_lo),
                    "q_0.975": _scalar(q_hi),
                }
            )
        return pd.DataFrame(rows).set_index("parameter")

    # --- Posterior sampling -------------------------------------------------

    def sample_posterior(self, n_samples=100):
        """Sample from the posterior distribution over parameters.

        Parameters
        ----------
        n_samples : int, default=100
            Number of posterior samples to draw.

        Returns
        -------
        samples : dict of str -> np.ndarray
            Parameter samples keyed by parameter name.
        """
        self.check_is_fitted()
        return self._sample_posterior(n_samples=n_samples)

    def _sample_posterior(self, n_samples=100):
        """Sample from posterior. Default uses distribution ``.sample()``."""
        posterior = self._get_posterior_params()
        return {
            name: dist.sample(n_samples) for name, dist in posterior.items()
        }

    # --- Sequential updating -----------------------------------------------

    def _update(self, X, y, C=None):
        """Bayesian sequential update — posterior becomes new prior.

        Default raises ``NotImplementedError``; subclasses with efficient
        conjugate or incremental updates should override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _update. "
            "Re-fit with all data or use an estimator that supports updates."
        )
