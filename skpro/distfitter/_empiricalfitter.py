"""Empirical distribution fitter."""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import pandas as pd

from skpro.distfitter.base import BaseDistFitter


class EmpiricalFitter(BaseDistFitter):
    """Fit an empirical distribution by wrapping data in an Empirical distribution.

    Converts the full sample into an empirical distribution.
    For the univariate case (empirical per variable), this simply wraps the
    data in an ``Empirical`` distribution.

    This is useful as a base component, e.g., for naive distribution fitting
    in ensemble or reduction strategies.

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.distfitter import EmpiricalFitter
    >>> X = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0])

    >>> fitter = EmpiricalFitter()
    >>> fitter.fit(X)
    EmpiricalFitter()
    >>> dist = fitter.proba()
    """

    _tags = {
        "authors": ["utsab345"],
    }

    def _fit(self, X, C=None):
        """Fit empirical distribution by storing the data.

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
        self.spl_ = pd.DataFrame(vals, columns=["spl"])
        return self

    def _proba(self):
        """Return fitted Empirical distribution.

        Returns
        -------
        dist : skpro Empirical distribution (scalar)
        """
        from skpro.distributions.empirical import Empirical

        return Empirical(spl=self.spl_)

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
