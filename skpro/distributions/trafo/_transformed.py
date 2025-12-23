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

    assume_monotonic : bool, optional, default = True
        whether to assume that the transform is monotonic, i.e., that
        the distribution is transformed in a way that preserves order of sample values.

    inverse_transform : callable, optional, default = None
        inverse function of ``transform``, if known.
        Must be applicable to array-likes of the same shape as ``self``.

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

    If the inverse is known, it can be given to ensure more methods are exact:
    >>> t = TransformedDistribution(
    ...     distribution=n,
    ...     transform=np.exp,
    ...     inverse_transform=np.log,
    ... )
    """

    _tags = {
        "capabilities:approx": [
            "pdfnorm",
            "mean",
            "var",
            "energy",
            "cdf",
        ],
        "capabilities:exact": ["ppf"],
        "distr:measuretype": "mixed",
        "distr:paramtype": "composite",
    }

    def __init__(
        self,
        distribution,
        transform,
        assume_monotonic=True,
        inverse_transform=None,
        index=None,
        columns=None,
    ):
        self.distribution = distribution
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.assume_monotonic = assume_monotonic

        self._is_scalar_dist = self.distribution.ndim == 0

        # determine index and columns
        if not self._is_scalar_dist:
            if index is None or columns is None:
                _example = self.distribution.sample()
                n_rows = _example.shape[0]
                n_cols = _example.shape[1]
                if index is None:
                    index = pd.RangeIndex(n_rows)
                if columns is None:
                    columns = pd.RangeIndex(n_cols)

        super().__init__(index=index, columns=columns)

        # transformed discret distributions are always discrete
        # (otherwise we only know that they are mixed)
        if distribution.get_tag("distr:measuretype") == "discrete":
            self.set_tags(**{"distr:measuretype": "discrete"})

        # if inverse_transform is given, we can do exact cdf
        # due to the formula F_g(x)(y) = F_X(g^-1(x))
        if self.inverse_transform is not None:
            self.set_tags(
                **{
                    "capabilities:exact": ["ppf", "cdf"],
                    "capabilities:approx": ["pdfnorm", "mean", "var", "energy"],
                }
            )

    def _iloc(self, rowidx=None, colidx=None):
        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._iat(rowidx, colidx)
        if is_scalar_notnone(rowidx):
            rowidx = pd.Index([rowidx])
        if is_scalar_notnone(colidx):
            colidx = pd.Index([colidx])

        def subset_not_none(idx, subs):
            if subs is not None:
                return idx.take(subs)
            else:
                return idx

        index_subset = subset_not_none(self.index, rowidx)
        columns_subset = subset_not_none(self.columns, colidx)

        distr = self.distribution.iloc[rowidx, colidx]

        # these parameters are manually subset
        # the other parameters are passed through
        POP_PARAMS = ["distribution", "index", "columns"]

        sk_distr_type = type(self)
        params_dict = self.get_params(deep=False)
        [params_dict.pop(param) for param in POP_PARAMS]

        return sk_distr_type(
            distribution=distr,
            index=index_subset,
            columns=columns_subset,
            **params_dict,
        )

    def _iat(self, rowidx=None, colidx=None):
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")
        self_subset = self.iloc[[rowidx], [colidx]]

        POP_PARAMS = ["distribution", "index", "columns"]

        sk_distr_type = type(self)
        params_dict = self.get_params(deep=False)
        [params_dict.pop(param) for param in POP_PARAMS]

        return sk_distr_type(
            distribution=self_subset.distribution.iat[0, 0],
            **params_dict,
        )

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Parameters
        ----------
        p : 2D np.ndarray, same shape as ``self``
            values to evaluate the ppf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            ppf values at the given points
        """
        if not self.assume_monotonic and self.inverse_transform is None:
            raise ValueError(
                "if inverse_transform is not given, "
                "ppf is implemented only for monotonic transforms, "
                "set `assume_monotonic=True` to use this method"
            )
        elif not self.assume_monotonic and self.inverse_transform is not None:
            return super().ppf(p)

        if self.ndim != 0:
            p = pd.DataFrame(p, index=self.index, columns=self.columns)

        trafo = self.transform

        inner_ppf = self.distribution.ppf(p)

        if self.ndim == 0:
            inner_ppf = pd.DataFrame([[float(inner_ppf)]])
        outer_ppf = trafo(inner_ppf)
        if self.ndim == 0:
            outer_ppf = _coerce_to_scalar(outer_ppf)

        if isinstance(outer_ppf, pd.DataFrame):
            # if the transform returns a DataFrame, we ensure the index and columns
            outer_ppf.index = self.index
            outer_ppf.columns = self.columns
        elif not self._is_scalar_dist:
            # if the transform returns a scalar or array, we  convert it to DataFrame
            outer_ppf = pd.DataFrame(outer_ppf, index=self.index, columns=self.columns)

        return outer_ppf

    def _cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the cdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            cdf values at the given points
        """
        if self.inverse_transform is None:
            return super()._cdf(x)

        inv_trafo = self.inverse_transform

        if self.ndim != 0:
            x = pd.DataFrame(x, index=self.index, columns=self.columns)
        else:
            x = pd.DataFrame([[float(x)]])

        inv_x = inv_trafo(x)
        if self.ndim == 0:
            inv_x = _coerce_to_scalar(inv_x)

        cdf_res = self.distribution.cdf(inv_x)

        if isinstance(cdf_res, pd.DataFrame):
            # if the transform returns a DataFrame, we ensure the index and columns
            cdf_res.index = self.index
            cdf_res.columns = self.columns
        elif not self._is_scalar_dist:
            # if the transform returns a scalar or array, we  convert it to DataFrame
            cdf_res = pd.DataFrame(cdf_res, index=self.index, columns=self.columns)

        return cdf_res

    def _sample(self, n_samples=None):
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

        # array case example with inverse transform
        n_array = Normal(mu=[[1, 2], [3, 4]], sigma=1, columns=pd.Index(["c", "d"]))
        params3 = {
            "distribution": n_array,
            "transform": lambda x: 2 * x,
            "inverse_transform": lambda x: x / 2,
            "index": pd.Index([1, 2]),
            "columns": pd.Index(["a", "b"]),  # this should override n_row.columns
        }

        # scalar case example with inverse transform
        params4 = {
            "distribution": n_scalar,
            "transform": np.exp,
            "inverse_transform": np.log,
        }

        return [params1, params2, params3, params4]


def is_scalar_notnone(obj):
    """Check if obj is scalar and not None."""
    return obj is not None and np.isscalar(obj)


def _coerce_to_scalar(x):
    """Coerce numpy or pd.DataFrame to numpy float."""
    if isinstance(x, pd.DataFrame):
        if x.shape != (1, 1):
            raise ValueError("input must be of shape (1, 1) to coerce to scalar")
        return x.iat[0, 0]
    elif isinstance(x, np.ndarray):
        if x.shape != (1, 1):
            raise ValueError("input must be of shape (1, 1) to coerce to scalar")
        return x[0, 0]
    else:
        return x
