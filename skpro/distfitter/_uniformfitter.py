"""Uniform distribution fitter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np

from skpro.distfitter.base import BaseDistFitter


class UniformFitter(BaseDistFitter):
    """Fit a Uniform distribution from the data range.

    Estimates ``lower`` and ``upper`` bounds of a Uniform distribution
    as the sample minimum and maximum, respectively.

    The data must contain at least two distinct values so that
    ``lower < upper``.

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.distfitter import UniformFitter
    >>> X = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> fitter = UniformFitter()
    >>> fitter.fit(X)
    UniformFitter()
    >>> dist = fitter.proba()
    """

    _tags = {
        "authors": ["patelchaitany"],
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X, C=None):
        """Fit Uniform distribution parameters from data.

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
        self.lower_ = float(np.min(vals))
        self.upper_ = float(np.max(vals))
        return self

    def _proba(self):
        """Return fitted Uniform distribution.

        Returns
        -------
        dist : skpro Uniform distribution (scalar)
        """
        from skpro.distributions.uniform import Uniform

        return Uniform(lower=self.lower_, upper=self.upper_)

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
