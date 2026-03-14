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
    dist_cls : class
        A distribution class from ``skpro.distributions``.
        Must accept keyword arguments for location and scale parameters.
    mean_name : str, optional (default="mu")
        Name of the distribution parameter that corresponds to the mean.
    std_name : str or None, optional (default=None)
        Name of the distribution parameter that corresponds to the
        standard deviation. If None, auto-detects by looking for
        ``"sigma"`` or ``"scale"`` in the ``__init__`` signature of
        ``dist_cls``.
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
    >>> fitter = MOMFitter(dist_cls=Normal, mean_name="mu", std_name="sigma")
    >>> fitter.fit(X)
    MOMFitter(...)
    >>> dist = fitter.proba()
    """

    _tags = {
        "authors": ["patelchaitany"],
        "reserved_params": ["dist_cls"],
    }

    def __init__(self, dist_cls, mean_name="mu", std_name=None, dist_params=None):
        self.dist_cls = dist_cls
        self.mean_name = mean_name
        self.std_name = std_name
        self.dist_params = dist_params

        super().__init__()

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Overrides base ``get_params`` to prevent deep recursion into
        ``dist_cls``, which is a class (not an instance) and would cause
        ``get_params`` to fail when called on it.

        Parameters
        ----------
        deep : bool, default=True
            If True, returns parameters of sub-objects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return super().get_params(deep=False)

    def _repr_html_(self):
        """HTML representation, overridden to avoid recursion into dist_cls."""
        return (
            f"<pre>{self.__class__.__name__}"
            f"(dist_cls={self.dist_cls.__name__}, "
            f"mean_name={self.mean_name!r}, "
            f"std_name={self.std_name!r}, "
            f"dist_params={self.dist_params!r})</pre>"
        )

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
        sig = inspect.signature(self.dist_cls.__init__)
        param_names = list(sig.parameters.keys())

        candidates = ["sigma", "scale"]
        for candidate in candidates:
            if candidate in param_names:
                return candidate

        raise ValueError(
            f"Could not auto-detect standard deviation parameter for "
            f"{self.dist_cls.__name__}. Signature has parameters "
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
        if self.dist_params is not None:
            params.update(self.dist_params)
        return self.dist_cls(**params)

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
        from skpro.distributions.normal import Normal

        params1 = {"dist_cls": Normal, "mean_name": "mu", "std_name": "sigma"}
        params2 = {"dist_cls": Normal, "mean_name": "mu"}
        params3 = {
            "dist_cls": Normal,
            "mean_name": "mu",
            "std_name": "sigma",
            "dist_params": {},
        }
        return [params1, params2, params3]
