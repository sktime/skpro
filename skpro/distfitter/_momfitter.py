"""Method of Moments distribution fitter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import inspect

import numpy as np

from skpro.distfitter.base import BaseDistFitter


class MOMFitter(BaseDistFitter):
    """Fit a distribution using method of moments (mean and standard deviation).

    Computes the sample mean and standard deviation from the data, then
    constructs the distribution given by ``dist_cls`` by plugging the mean
    into the parameter named ``mean_name`` and the standard deviation into
    the parameter named ``std_name``.

    Parameters
    ----------
    dist : skpro distribution instance or class
        A distribution instance or class from ``skpro.distributions``.
        Must accept keyword arguments for location and scale parameters.
    mean_name : str, optional (default="mu")
        Name of the distribution parameter that corresponds to the mean.
    std_name : str or None, optional (default=None)
        Name of the distribution parameter that corresponds to the
        standard deviation. If None, auto-detects by looking for
        ``"sigma"`` or ``"scale"`` in the ``__init__`` signature of ``dist``.
    dist_params : dict or None, optional (default=None)
        Additional fixed parameters to pass to the distribution constructor.
        These are merged with the estimated mean and std parameters when
        constructing the distribution in ``proba()``. Useful for distributions
        that require extra parameters beyond location and scale, e.g.,
        ``{"l_trunc": -3.0, "r_trunc": 3.0}`` for ``TruncatedNormal``,
        or ``{"df": 5}`` for ``Student-t``.

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.distfitter import MOMFitter
    >>> from skpro.distributions.normal import Normal
    >>> X = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> fitter = MOMFitter(dist=Normal, mean_name="mu", std_name="sigma")
    >>> fitter.fit(X)
    MOMFitter(...)
    >>> dist = fitter.proba()

    Using Laplace distribution (uses "scale" instead of "sigma"):

    >>> from skpro.distributions.laplace import Laplace
    >>> fitter = MOMFitter(dist=Laplace, mean_name="mu", std_name="scale")
    >>> fitter.fit(X)
    MOMFitter(...)

    Using Student-t with extra parameters via dist_params:

    >>> from skpro.distributions.t import TDistribution
    >>> fitter = MOMFitter(
    ...     dist=TDistribution, mean_name="mu", std_name="sigma",
    ...     dist_params={"df": 5},
    ... )
    >>> fitter.fit(X)
    MOMFitter(...)
    """

    _tags = {
        "authors": ["patelchaitany"],
        "reserved_params": ["dist"],
    }

    def __init__(self, dist, mean_name="mu", std_name=None, dist_params=None):
        self.dist = dist
        self.mean_name = mean_name
        self.std_name = std_name
        self.dist_params = dist_params

        super().__init__()

        if dist_params is None:
            self._dist_params = {}
        else:
            self._dist_params = dist_params

    def _fit(self, X, C=None):
        """Fit distribution parameters using method of moments.

        Parameters
        ----------
        X : pandas DataFrame
            Data to fit the distribution to.
        C : ignored

        Returns
        -------
        self : reference to self
        """
        vals = X.values.ravel()
        self.mean_ = float(np.mean(vals))
        self.std_ = float(np.std(vals, ddof=1))

        if self.std_name is not None:
            self._std_name_resolved = self.std_name
        else:
            self._std_name_resolved = self._resolve_std_name()

        return self

    def _resolve_std_name(self):
        """Auto-detect the standard deviation parameter name from dist_cls.

        Inspects the ``__init__`` signature of ``dist_cls`` and looks for
        known parameter names: ``"sigma"``, ``"scale"``.

        Returns
        -------
        str
            Resolved parameter name.

        Raises
        ------
        ValueError
            If no known standard deviation parameter is found.
        """
        param_names = self.dist.get_param_names()  # list(sig.parameters.keys())

        candidates = ["sigma", "scale"]
        for candidate in candidates:
            if candidate in param_names:
                return candidate

        raise ValueError(
            f"Could not auto-detect standard deviation parameter for "
            f"{self._dist.__class__.__name__}. Signature has parameters "
            f"{param_names}. Please set std_name explicitly to one of "
            f"these."
        )

    def _proba(self):
        """Return fitted distribution.

        Returns
        -------
        dist : skpro BaseDistribution (scalar)
        """
        params = {
            self.mean_name: self.mean_,
            self._std_name_resolved: self.std_,
        }
        params.update(self._dist_params)

        if inspect.isclass(self.dist):
            dist_with_params = self.dist(**params)
        else:
            dist_with_params = self.dist.clone().set_params(**params)

        return dist_with_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from skpro.distributions.laplace import Laplace
        from skpro.distributions.normal import Normal
        from skpro.distributions.t import TDistribution

        params1 = {"dist": Normal, "mean_name": "mu", "std_name": "sigma"}
        params2 = {"dist": Laplace, "mean_name": "mu", "std_name": "scale"}
        params3 = {"dist": Laplace, "mean_name": "mu"}
        params4 = {
            "dist": TDistribution,
            "mean_name": "mu",
            "std_name": "sigma",
            "dist_params": {"df": 5},
        }
        params5 = {"dist": Laplace(mu=0.0, scale=1.0), "mean_name": "mu"}
        params6 = {
            "dist": TDistribution(df=5, mu=0.0, sigma=1.0),
            "mean_name": "mu",
            "std_name": "sigma",
        }
        return [params1, params2, params3, params4, params5, params6]
