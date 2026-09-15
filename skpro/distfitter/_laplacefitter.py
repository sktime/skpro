"""Laplace distribution fitter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np

from skpro.distfitter.base import BaseDistFitter


class LaplaceFitter(BaseDistFitter):
    r"""Fit a Laplace distribution using robust estimators.

    Estimates the location parameter :math:`\mu` as the sample median
    and the scale parameter :math:`b` from the median absolute deviation
    (MAD), scaled by :math:`1 / \ln 2` to match the MLE relationship:

    .. math:: \hat{b} = \frac{\text{MAD}}{\ln 2}

    where :math:`\text{MAD} = \text{median}(|x_i - \hat{\mu}|)`.

    Using the median and MAD makes this fitter more robust to outliers
    than a method-of-moments estimator based on mean and standard
    deviation.

    Parameters
    ----------
    method : str, optional (default="robust")
        Estimation method:

        - ``"robust"`` : median for location, MAD / ln(2) for scale.
        - ``"MLE"`` : median for location, mean absolute deviation from
          median for scale (the classical MLE for Laplace).

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.distfitter import LaplaceFitter
    >>> X = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> fitter = LaplaceFitter()
    >>> fitter.fit(X)
    LaplaceFitter()
    >>> dist = fitter.proba()
    """

    _tags = {
        "authors": ["patelchaitany"],
    }

    VALID_METHODS = ("robust", "MLE")

    def __init__(self, method="robust"):
        self.method = method
        super().__init__()

    def _fit(self, X, C=None):
        """Fit Laplace distribution parameters from data.

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
        self.mu_ = float(np.median(vals))

        abs_devs = np.abs(vals - self.mu_)

        if method == "robust":
            mad = float(np.median(abs_devs))
            self.scale_ = mad / np.log(2)
        elif method == "MLE":
            self.scale_ = float(np.mean(abs_devs))

        return self

    def _proba(self):
        """Return fitted Laplace distribution.

        Returns
        -------
        dist : skpro Laplace distribution (scalar)
        """
        from skpro.distributions.laplace import Laplace

        return Laplace(mu=self.mu_, scale=self.scale_)

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
        params1 = {"method": "robust"}
        params2 = {"method": "MLE"}
        return [params1, params2]
