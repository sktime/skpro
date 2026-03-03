# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Temporal Normal/Gaussian probability distribution with time-varying parameters."""

import numpy as np
import pandas as pd

from skpro.distributions.normal import Normal


class TemporalNormal(Normal):
    r"""Temporal Normal distribution with time-dependent parameters.

    This distribution extends the Normal distribution to explicitly handle
    time-varying parameters. The mean :math:`\mu(t)` and standard deviation
    :math:`\sigma(t)` can vary over time, allowing the representation of
    temporal correlation in the data.

    The temporal normal distribution is parametrized by time-dependent mean
    :math:`\mu(t)` and time-dependent standard deviation :math:`\sigma(t)`,
    such that at each time point :math:`t`, the pdf is:

    .. math:: f(x|t) = \frac{1}{\sigma(t) \sqrt{2\pi}} \exp\left(-\frac{(x - \mu(t))^2}{2\sigma(t)^2}\right)

    The time-dependent mean :math:`\mu(t)` is represented by the parameter ``mu``,
    and the time-dependent standard deviation :math:`\sigma(t)` by the parameter ``sigma``.

    This distribution is particularly useful for:
    * Time series forecasting with uncertainty quantification
    * Modeling data with temporal trends in mean and variance
    * Representing non-stationary processes

    Note
    ----
    This implementation models time-varying marginal normals only.
    It does not model temporal covariance/correlation between time points.

    Parameters
    ----------
    mu : float, array of float (1D or 2D), pd.Series, or pd.DataFrame
        time-dependent mean of the normal distribution.
        If pd.Series or pd.DataFrame, the index represents time points.
    sigma : float, array of float (1D or 2D), pd.Series, or pd.DataFrame, must be positive
        time-dependent standard deviation of the normal distribution.
        If pd.Series or pd.DataFrame, the index should match ``mu``.
    index : pd.Index, optional, default = RangeIndex
        time index for the distribution. If ``mu`` or ``sigma`` are pandas objects
        with an index, this will be inferred from them.
    columns : pd.Index, optional, default = RangeIndex
        column index for multivariate case

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skpro.distributions.temporal_normal import TemporalNormal

    Example 1: Time-varying mean with constant variance
    >>> time_index = pd.date_range('2024-01-01', periods=10, freq='D')
    >>> mu_t = pd.Series(np.linspace(0, 10, 10), index=time_index)
    >>> dist = TemporalNormal(mu=mu_t, sigma=1.0)

    Example 2: Both mean and variance varying over time
    >>> mu_t = pd.Series([0, 1, 2, 3, 4], index=pd.RangeIndex(5))
    >>> sigma_t = pd.Series([0.5, 0.7, 1.0, 1.2, 1.5], index=pd.RangeIndex(5))
    >>> dist = TemporalNormal(mu=mu_t, sigma=sigma_t)

    Example 3: Multivariate time-varying distribution
    >>> time_index = pd.RangeIndex(5)
    >>> mu_t = pd.DataFrame([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], index=time_index)
    >>> sigma_t = pd.DataFrame([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2], [1.3, 1.4]], index=time_index)
    >>> dist = TemporalNormal(mu=mu_t, sigma=sigma_t)

    Example 4: Array-based specification (as in Normal)
    >>> dist = TemporalNormal(mu=[[0, 1], [2, 3], [4, 5]], sigma=[[1, 1], [1.5, 1.5], [2, 2]])
    """  # noqa E501

    _tags = {
        # packaging info
        # --------------
        "authors": ["arnavk23"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma

        # Convert pandas Series to DataFrame for proper time-series orientation
        # Series are typically treated as row vectors (1, n) but for time series
        # we want column vectors (n, 1) where n is the number of time points
        mu_inner = mu
        sigma_inner = sigma
        if isinstance(mu, pd.Series):
            mu_inner = mu.to_frame()
        if isinstance(sigma, pd.Series):
            sigma_inner = sigma.to_frame()

        self._mu_inner = mu_inner
        self._sigma_inner = sigma_inner

        # Handle pandas DataFrame inputs with time index
        if isinstance(mu_inner, pd.DataFrame) and index is None:
            index = mu_inner.index
        if isinstance(sigma_inner, pd.DataFrame) and index is None:
            index = sigma_inner.index

        # Call parent Normal class __init__
        # This will handle all the broadcasting and parameter setup
        super().__init__(mu=mu_inner, sigma=sigma_inner, index=index, columns=columns)

        self.mu = mu
        self.sigma = sigma

    def _get_dist_params(self):
        """Return internal broadcast-ready distribution parameters."""
        return {"mu": self._mu_inner, "sigma": self._sigma_inner}

    def mean_at_time(self, t):
        """Return the mean at a specific time point.

        Parameters
        ----------
        t : int or time index
            Time point to query

        Returns
        -------
        float or array
            Mean value(s) at time t
        """
        if self.ndim == 0:
            return self.mu

        # Get the broadcasted parameters
        mu_bc = self._bc_params.get("mu", self.mu)

        if isinstance(self.index, pd.DatetimeIndex) or hasattr(self.index, "get_loc"):
            try:
                loc = self.index.get_loc(t)
                if isinstance(mu_bc, np.ndarray) and mu_bc.ndim >= 2:
                    return mu_bc[loc, :]
                elif hasattr(mu_bc, "__getitem__"):
                    return mu_bc[loc]
                else:
                    return mu_bc
            except (KeyError, TypeError):
                raise ValueError(f"Time point {t} not found in distribution index")
        else:
            if isinstance(t, int) and 0 <= t < len(self.index):
                if isinstance(mu_bc, np.ndarray) and mu_bc.ndim >= 2:
                    return mu_bc[t, :]
                elif hasattr(mu_bc, "__getitem__"):
                    return mu_bc[t]
                else:
                    return mu_bc
            else:
                raise ValueError(f"Time index {t} out of range")

    def var_at_time(self, t):
        """Return the variance at a specific time point.

        Parameters
        ----------
        t : int or time index
            Time point to query

        Returns
        -------
        float or array
            Variance value(s) at time t
        """
        if self.ndim == 0:
            return self.sigma**2

        # Get the broadcasted parameters
        sigma_bc = self._bc_params.get("sigma", self.sigma)

        if isinstance(self.index, pd.DatetimeIndex) or hasattr(self.index, "get_loc"):
            try:
                loc = self.index.get_loc(t)
                if isinstance(sigma_bc, np.ndarray) and sigma_bc.ndim >= 2:
                    sigma_t = sigma_bc[loc, :]
                elif hasattr(sigma_bc, "__getitem__"):
                    sigma_t = sigma_bc[loc]
                else:
                    sigma_t = sigma_bc
                return sigma_t**2
            except (KeyError, TypeError):
                raise ValueError(f"Time point {t} not found in distribution index")
        else:
            if isinstance(t, int) and 0 <= t < len(self.index):
                if isinstance(sigma_bc, np.ndarray) and sigma_bc.ndim >= 2:
                    sigma_t = sigma_bc[t, :]
                elif hasattr(sigma_bc, "__getitem__"):
                    sigma_t = sigma_bc[t]
                else:
                    sigma_t = sigma_bc
                return sigma_t**2
            else:
                raise ValueError(f"Time index {t} out of range")

    def std_at_time(self, t):
        """Return the standard deviation at a specific time point.

        Parameters
        ----------
        t : int or time index
            Time point to query

        Returns
        -------
        float or array
            Standard deviation value(s) at time t
        """
        if self.ndim == 0:
            return self.sigma

        # Get the broadcasted parameters
        sigma_bc = self._bc_params.get("sigma", self.sigma)

        if isinstance(self.index, pd.DatetimeIndex) or hasattr(self.index, "get_loc"):
            try:
                loc = self.index.get_loc(t)
                if isinstance(sigma_bc, np.ndarray) and sigma_bc.ndim >= 2:
                    return sigma_bc[loc, :]
                elif hasattr(sigma_bc, "__getitem__"):
                    return sigma_bc[loc]
                else:
                    return sigma_bc
            except (KeyError, TypeError):
                raise ValueError(f"Time point {t} not found in distribution index")
        else:
            if isinstance(t, int) and 0 <= t < len(self.index):
                if isinstance(sigma_bc, np.ndarray) and sigma_bc.ndim >= 2:
                    return sigma_bc[t, :]
                elif hasattr(sigma_bc, "__getitem__"):
                    return sigma_bc[t]
                else:
                    return sigma_bc
            else:
                raise ValueError(f"Time index {t} out of range")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # Time-series case with pandas index
        params1 = {
            "mu": pd.Series([0, 1, 2, 3, 4]),
            "sigma": pd.Series([0.5, 0.7, 1.0, 1.2, 1.5]),
        }

        # Array case with explicit time index
        params2 = {
            "mu": [[0, 1], [2, 3], [4, 5]],
            "sigma": [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
        }

        # Scalar variance with time-varying mean
        params3 = {
            "mu": pd.Series(
                [0, 2, 4, 6], index=pd.date_range("2024-01-01", periods=4, freq="D")
            ),
            "sigma": 1.0,
        }

        return [params1, params2, params3]
