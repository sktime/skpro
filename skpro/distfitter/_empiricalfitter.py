"""Empirical distribution fitter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.distfitter.base import BaseDistFitter


class EmpiricalFitter(BaseDistFitter):
    """Fit an empirical distribution by wrapping data in an Empirical distribution.

    Converts the full sample into an empirical distribution.
    For the univariate case (empirical per variable), this simply wraps the
    data in an ``Empirical`` distribution.

    This is useful as a base component, e.g., for naive distribution fitting
    in ensemble or reduction strategies.

    Parameters
    ----------
    time_indep : bool, optional (default=True)
        If True, the empirical distribution will sample individual instance
        indices independently. If False, it will sample entire instances.
        Passed through to ``Empirical``.

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
        "authors": ["fkiraly"],
    }

    def __init__(self, time_indep=True):
        self.time_indep = time_indep

        super().__init__()

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
        import pandas as pd

        if X.shape[1] > 1:
            n = len(X)
            idx = pd.MultiIndex.from_arrays(
                [range(n), [0] * n], names=["sample", None]
            )
            spl = X.copy()
            spl.index = idx
        else:
            spl = X
        self.spl_ = spl
        return self

    def _proba(self):
        """Return fitted Empirical distribution.

        Returns
        -------
        dist : skpro Empirical distribution (scalar)
        """
        from skpro.distributions.empirical import Empirical

        return Empirical(spl=self.spl_, time_indep=self.time_indep)

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
        params2 = {"time_indep": False}
        return [params1, params2]
