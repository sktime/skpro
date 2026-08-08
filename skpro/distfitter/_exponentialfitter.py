"""Exponential distribution fitter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np

from skpro.distfitter.base import BaseDistFitter


class ExponentialFitter(BaseDistFitter):
    r"""Fit an Exponential distribution using the sample mean.

    Estimates the rate parameter :math:`\lambda` of an Exponential
    distribution as the reciprocal of the sample mean:
    :math:`\hat{\lambda} = n / \sum x_i`.

    The data passed to ``fit`` must be strictly positive.

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.distfitter import ExponentialFitter
    >>> X = pd.DataFrame([0.5, 1.0, 1.5, 2.0, 2.5])
    >>> fitter = ExponentialFitter()
    >>> fitter.fit(X)
    ExponentialFitter()
    >>> dist = fitter.proba()
    """

    _tags = {
        "authors": ["patelchaitany"],
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X, C=None):
        """Fit Exponential distribution parameters from data.

        Parameters
        ----------
        X : pandas DataFrame
            Data to fit the distribution to. Must be strictly positive.
        C : ignored

        Returns
        -------
        self : reference to self
        """
        vals = X.values.ravel()
        self.rate_ = float(len(vals) / np.sum(vals))
        return self

    def _proba(self):
        """Return fitted Exponential distribution.

        Returns
        -------
        dist : skpro Exponential distribution (scalar)
        """
        from skpro.distributions.exponential import Exponential

        return Exponential(rate=self.rate_)

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
        return [{}]
