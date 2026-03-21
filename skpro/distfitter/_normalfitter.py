"""Normal distribution fitter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np

from skpro.distfitter.base import BaseDistFitter


class NormalFitter(BaseDistFitter):
    r"""Fit a Normal distribution using sample mean and standard deviation.

    Estimates the parameters :math:`\mu` and :math:`\sigma` of a Normal
    distribution from the sample mean and sample standard deviation of the
    data passed to ``fit``.

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.distfitter import NormalFitter
    >>> X = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> fitter = NormalFitter()
    >>> fitter.fit(X)
    NormalFitter()
    >>> dist = fitter.proba()
    """

    _tags = {
        "authors": ["patelchaitany"],
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X, C=None):
        """Fit Normal distribution parameters from data.

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
        self.mu_ = float(np.mean(vals))
        self.sigma_ = float(np.std(vals, ddof=1))
        return self

    def _proba(self):
        """Return fitted Normal distribution.

        Returns
        -------
        dist : skpro Normal distribution (scalar)
        """
        from skpro.distributions.normal import Normal

        return Normal(mu=self.mu_, sigma=self.sigma_)

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
        params1 = {}
        params2 = {}
        return [params1, params2]
