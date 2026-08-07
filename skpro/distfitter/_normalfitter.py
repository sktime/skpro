"""Normal distribution fitter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np

from skpro.distfitter.base import BaseDistFitter


class NormalFitter(BaseDistFitter):
    r"""Fit a Normal distribution using sample mean and standard deviation.

    Estimates the parameters :math:`\mu` and :math:`\sigma` of a Normal
    distribution from the sample mean and sample standard deviation of the
    data passed to ``fit``.

    Parameters
    ----------
    method : str, optional (default="unbiased")
        Estimation method for the standard deviation:

        - ``"unbiased"`` : sample standard deviation with Bessel's correction
          (denominator :math:`N - 1`).
        - ``"MLE"`` : maximum likelihood estimate (denominator :math:`N`).
        - ``"shrinkage"`` : linear shrinkage of the sample variance towards
          ``shrinkage_target``. The shrinkage intensity is controlled by
          ``shrinkage_alpha``.

    shrinkage_target : float, optional (default=1.0)
        Target value for the variance when ``method="shrinkage"``.
        Ignored for other methods.
    shrinkage_alpha : float, optional (default=0.5)
        Shrinkage intensity in :math:`[0, 1]`. The shrunk variance is computed
        as :math:`(1 - \alpha) \cdot s^2 + \alpha \cdot \text{target}`, where
        :math:`s^2` is the unbiased sample variance. ``0`` means no shrinkage
        (equivalent to ``"unbiased"``), ``1`` means full shrinkage to target.
        Ignored for other methods.

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.distfitter import NormalFitter
    >>> X = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0])

    Unbiased estimator (default, denominator N-1):

    >>> fitter = NormalFitter()
    >>> fitter.fit(X)
    NormalFitter()
    >>> dist = fitter.proba()

    Maximum likelihood estimator (denominator N):

    >>> fitter_mle = NormalFitter(method="MLE")
    >>> fitter_mle.fit(X)
    NormalFitter(method='MLE')
    >>> dist_mle = fitter_mle.proba()

    Shrinkage estimator (shrink variance towards target):

    >>> fitter_shr = NormalFitter(
    ...     method="shrinkage", shrinkage_target=1.0, shrinkage_alpha=0.3,
    ... )
    >>> fitter_shr.fit(X)
    NormalFitter(method='shrinkage', shrinkage_alpha=0.3)
    >>> dist_shr = fitter_shr.proba()
    """

    _tags = {
        "authors": ["patelchaitany"],
    }

    VALID_METHODS = ("unbiased", "MLE", "shrinkage")

    def __init__(self, method="unbiased", shrinkage_target=1.0, shrinkage_alpha=0.5):
        self.method = method
        self.shrinkage_target = shrinkage_target
        self.shrinkage_alpha = shrinkage_alpha

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
        method = self.method
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Must be one of {self.VALID_METHODS}."
            )

        vals = X.values.ravel()
        self.mu_ = float(np.mean(vals))

        if method == "unbiased":
            self.sigma_ = float(np.std(vals, ddof=1))
        elif method == "MLE":
            self.sigma_ = float(np.std(vals, ddof=0))
        elif method == "shrinkage":
            alpha = self.shrinkage_alpha
            target = self.shrinkage_target
            sample_var = float(np.var(vals, ddof=1))
            shrunk_var = (1 - alpha) * sample_var + alpha * target
            self.sigma_ = float(np.sqrt(shrunk_var))

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
        params1 = {"method": "unbiased"}
        params2 = {"method": "MLE"}
        params3 = {
            "method": "shrinkage",
            "shrinkage_target": 2.0,
            "shrinkage_alpha": 0.3,
        }
        return [params1, params2, params3]
