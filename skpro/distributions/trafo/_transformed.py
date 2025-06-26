# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Transformed distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class TransformedDistribution(BaseDistribution):
    r"""Distribution transformed by an entry-wise function.

    Constructed with:

    * ``distribution``: a skpro distribution object, which is transformed
    * ``transform``: a function that is applied to the distribution.
      This can be entry-wise, or a ``pandas.DataFrame`` to ``pandas.DataFrame``
      function that can be applied to samples from ``distribution``.

    Parameters
    ----------
    distribution : skpro distribution - must be same shape as ``self``

    transform : callable
        function that is applied to the distribution, must be applicable
        to array-likes of the same shape as ``self``.

    assume_monotonic : bool, optional, default = False
        whether to assume that the transform is monotonic, i.e., that
        the distribution is transformed in a way that preserves order of sample values.

    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skpro.distributions.trafo import TransformedDistribution
    >>> from skpro.distributions import Normal
    >>>
    >>> n = Normal(mu=0, sigma=1)
    >>> # transform the distribution by taking the exponential
    >>> t = TransformedDistribution(distribution=n, transform=np.exp)
    """

    _tags = {
        "capabilities:approx": [
            "pdfnorm",
            "mean",
            "var",
            "energy",
            "cdf",
            "ppf",
        ],
        "capabilities:exact": [],
        "distr:measuretype": "discrete",
        "distr:paramtype": "composite",
    }

    def __init__(
        self,
        distribution,
        transform,
        assume_monotonic=False,
        index=None,
        columns=None,
    ):
        self.distribution = distribution
        self.transform = transform
        self.assume_monotonic = assume_monotonic

        self._is_scalar_dist = self.distribution.ndim == 0

        # determine index and columns
        if not self._is_scalar_dist:
            if index is None or columns is None:
                _example = self.distribution.sample()
                if isinstance(_example, pd.DataFrame):
                    if index is None:
                        index = _example.index
                    if columns is None:
                        columns = _example.columns

        super().__init__(index=index, columns=columns)

    def sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None
            number of samples to draw from the distribution

        Returns
        -------
        pd.DataFrame
            samples from the distribution

            * if ``n_samples`` is ``None``:
            returns a sample that contains a single sample from ``self``,
            in ``pd.DataFrame`` mtype format convention, with ``index`` and ``columns``
            as ``self``
            * if n_samples is ``int``:
            returns a ``pd.DataFrame`` that contains ``n_samples`` i.i.d.
            samples from ``self``, in ``pd-multiindex`` mtype format convention,
            with same ``columns`` as ``self``, and row ``MultiIndex`` that is product
            of ``RangeIndex(n_samples)`` and ``self.index``
        """
        # else we sample manually, this will be less efficient due to loops
        if n_samples is None:
            n = 1
        else:
            n = n_samples

        trafo = self.transform

        samples = []

        for _ in range(n):
            new_sample = trafo(self.distribution.sample())
            if not self._is_scalar_dist:
                if not isinstance(new_sample, pd.DataFrame):
                    new_sample = pd.DataFrame(
                        new_sample, index=self.index, columns=self.columns
                    )
                else:
                    new_sample.index = self.index
                    new_sample.columns = self.columns
            samples.append(new_sample)

        if n_samples is None:
            return samples[0]

        if self._is_scalar_dist:
            return pd.DataFrame(samples)

        # if n_samples is int, we return a DataFrame with MultiIndex
        res = pd.concat(samples, axis=0, keys=range(n))
        return res

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from skpro.distributions import Normal

        n_scalar = Normal(mu=0, sigma=1)
        # scalar case example
        params1 = {
            "distribution": n_scalar,
            "transform": np.exp,
        }

        # array case example
        n_array = Normal(mu=[[1, 2], [3, 4]], sigma=1, columns=pd.Index(["c", "d"]))
        params2 = {
            "distribution": n_array,
            "transform": np.exp,
            "index": pd.Index([1, 2]),
            "columns": pd.Index(["a", "b"]),  # this should override n_row.columns
        }

        return [params1, params2]
