# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Transformed distribution."""

__author__ = ["fkiraly"]

from functools import partial

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

    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> from skpro.distributions.trafo import TransformedDistribution
    >>> from skpro.distributions import Normal
    >>>
    >>> n = Normal(mu=0, sigma=1)
    >>> # transform the distribution by taking the exponential
    >>> ft = FunctionTransformer(func=np.log, inverse_func=np.exp)
    >>> t = TransformedDistribution(distribution=n, transformer=ft)
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
        "distr:measuretype": "discrete",
        "distr:paramtype": "composite",
    }

    def __init__(
        self,
        distribution,
        transformer,
        assume_monotonic=True,
        index=None,
        columns=None,
        numerical_diff=True,
    ):
        self.distribution = distribution
        self.transformer = transformer
        self.assume_monotonic = assume_monotonic
        self.numerical_diff = numerical_diff

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

    def log_pdf(self, x: pd.DataFrame):
        r"""Logarithmic probability density function.

        Parameters
        ----------
        x : pd.DataFrame, same shape as ``self``
            points where the log-pdf is evaluated

        Returns
        -------
        pd.DataFrame, same shape as ``self``
            log-pdf values at the given points
        """
        dist = self.distribution
        transformer = self.transformer
        x_ = transformer.transform(x)

        if hasattr(dist, "_distribution_attr"):
            obj = getattr(dist, dist._distribution_attr)
            args, kwds = dist._get_scipy_param()
            log_pdf = partial(obj.logpdf, *args, **kwds)
        else:
            log_pdf = dist.log_pdf

        jac = self._jacobian(x)
        return log_pdf(x_) - np.log(jac).reshape(x.shape)

    def _jacobian(self, x: pd.DataFrame):
        """Compute the Jacobian matrix of the transform at x.

        Parameters
        ----------
        x : pd.DataFrame, same shape as ``self``
            points where the Jacobian is evaluated

        Returns
        -------
        np.ndarray of shape (n, n, m, m) where (n, m) is the shape of ``self``
            Jacobian matrices at the given points
        """
        transformer = self.transformer

        if hasattr(transformer, "scale_"):
            # if the transform has scale_, we can compute the Jacobian
            jac = abs(1 / transformer.scale_)
            jac = np.ones_like(x) * jac
            # TODO: return x-shaped array
            return jac
        elif self.numerical_diff:
            # implement numpy differentiation here
            x_t_np = transformer.transform(x).values.reshape(-1)
            x_np = x.values.reshape(-1)
            grad = ordered_gradient(x_np, x_t_np)
            jac = np.abs(grad)
            return jac
        else:
            raise NotImplementedError(
                "Jacobian computation not implemented for this transformer, "
                "set `numerical_diff=True` to use numerical differentiation or "
                "use a transformer with `scale_` attribute, e.g., MinMaxScaler"
            )

    def _iloc(self, rowidx=None, colidx=None):
        distr = self.distribution.iloc[rowidx, colidx]

        if rowidx is not None:
            new_index = self.index[rowidx]
        else:
            new_index = self.index

        if colidx is not None:
            new_columns = self.columns[colidx]
        else:
            new_columns = self.columns

        cls = type(self)
        return cls(
            distribution=distr,
            transformer=self.transformer,
            assume_monotonic=self.assume_monotonic,
            index=new_index,
            columns=new_columns,
        )

    def _iat(self, rowidx=None, colidx=None):
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")
        self_subset = self.iloc[[rowidx], [colidx]]
        return type(self)(distribution=self_subset.distribution.iat[0, 0])

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
        if not self.assume_monotonic:
            raise ValueError(
                "ppf is implemented only for monotonic transforms, "
                "set `assume_monotonic=True` to use this method"
            )

        trafo = self.transformer.inverse_transform

        inner_ppf = self.distribution.ppf(p)
        outer_ppf = trafo(inner_ppf)

        if isinstance(outer_ppf, pd.DataFrame):
            # if the transform returns a DataFrame, we ensure the index and columns
            outer_ppf.index = self.index
            outer_ppf.columns = self.columns
        elif not self._is_scalar_dist:
            # if the transform returns a scalar or array, we  convert it to DataFrame
            outer_ppf = pd.DataFrame(outer_ppf, index=self.index, columns=self.columns)

        return outer_ppf

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

        trafo = self.transformer.inverse_transform

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
        from sklearn.preprocessing import FunctionTransformer

        from skpro.distributions import Normal

        ft = FunctionTransformer(func=np.log, inverse_func=np.exp)

        n_scalar = Normal(mu=0, sigma=1)
        # scalar case example
        params1 = {
            "distribution": n_scalar,
            "transformer": ft,
        }

        # array case example
        n_array = Normal(mu=[[1, 2], [3, 4]], sigma=1, columns=pd.Index(["c", "d"]))
        params2 = {
            "distribution": n_array,
            "transformer": ft,
            "index": pd.Index([1, 2]),
            "columns": pd.Index(["a", "b"]),  # this should override n_row.columns
        }

        return [params1, params2]


def is_scalar_notnone(obj):
    """Check if obj is scalar and not None."""
    return obj is not None and np.isscalar(obj)


def ordered_gradient(f, x):
    """Compute np.gradient(f, x) but safely handles when not already unsorted by x."""
    # Ensure numpy arrays
    f = np.asarray(f)
    x = np.asarray(x)

    # Sort x and reorder f accordingly
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    f_sorted = f[sort_idx]

    # Compute gradient
    grad_sorted = np.gradient(f_sorted, x_sorted)

    # Map back to original order
    grad = np.empty_like(grad_sorted)
    grad[sort_idx] = grad_sorted

    return grad
