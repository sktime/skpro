# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""I.i.d. sample from a distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class IID(BaseDistribution):
    r"""An i.i.d. sample of a given distribution.

    Constructed with a scalar or row distribution.

    Broadcasts to an i.i.d. sample based on index and columns.

    If ``distribution`` is scalar, broadcasts across index and columns.

    If ``distribution`` is a row distribution, ``columns`` must be
    same number of rows if passed; broadcasts across index.

    Parameters
    ----------
    distribution : skpro distribution - scalar, or shape (1, d)
        mean of the normal distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> import pandas as pd
    >>> from skpro.distributions.compose import IID
    >>> from skpro.distributions.normal import Normal

    >>> n = Normal(mu=0, sigma=1)
    >>> index = pd.Index([1, 2, 4, 6])
    >>> n_iid = IID(n, index=index)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "composite",
    }

    def __init__(self, distribution, index=None, columns=None):
        self.distribution = distribution

        dist_scalar = distribution.ndim == 0
        dist_cols = distribution.columns

        # set _bc_cols - do we broadcast rows?
        if dist_scalar:
            self._bc_cols = True
        else:  # not dist_scalar
            if distribution.shape[1] == 1 and columns is not None:
                self._bc_cols = True
            else:
                self._bc_cols = False

        # what is the index of self?
        if not dist_scalar and index is None:
            index = distribution.index
            assert len(index) == 1
        if dist_scalar and index is None:
            index = pd.RangeIndex(1)
        # else index is what was passed

        # what are columns of self?
        if dist_scalar and columns is None:
            columns = pd.RangeIndex(1)
        if not dist_scalar:
            if columns is None:
                columns = dist_cols
            if not len(dist_cols) == 1 and columns is not None:
                assert len(dist_cols) == len(columns)

        super().__init__(index=index, columns=columns)

        tags_to_clone = [
            "distr:measuretype",
            "capabilities:exact",
            "capabilities:approx",
        ]
        self.clone_tags(distribution, tags_to_clone)

    # TODO - use outer product once implemented, see issue #341
    # switch the method out to _broadcast_iid_future
    def _broadcast_iid(self, method, **kwargs):
        if self.ndim == 0:
            return getattr(self.distribution, method)(**kwargs)

        def _to_numpy(v):
            if isinstance(v, pd.DataFrame):
                return v.values
            return v

        kwargs = {k: _to_numpy(v) for k, v in kwargs.items()}

        # methods that return a single column always
        METHODS_TO_1D = ["energy"]

        if len(kwargs) == 0:
            one_dist_res = getattr(self.distribution, method)()
            if isinstance(one_dist_res, pd.DataFrame):
                one_dist_res = one_dist_res.values
            res = np.broadcast_to(one_dist_res, self.shape)
        else:
            if self._bc_cols:
                res = np.zeros(self.shape)
                for i, j in np.ndindex(self.shape):
                    kwargs_ij = {k: v[i, j] for k, v in kwargs.items()}
                    res[i, j] = getattr(self.distribution, method)(**kwargs_ij)
            elif method not in METHODS_TO_1D:
                res = np.zeros(self.shape)
                for i in range(self.shape[0]):
                    kwargs_i = {k: v[i] for k, v in kwargs.items()}
                    res[i] = getattr(self.distribution, method)(**kwargs_i)
            else:
                res = np.zeros((self.shape[0], 1))
                for i in range(self.shape[0]):
                    kwargs_i = {k: v[i] for k, v in kwargs.items()}
                    res[i] = getattr(self.distribution, method)(**kwargs_i)

        if method in METHODS_TO_1D and res.shape[1] > 1:
            res = np.mean(res, axis=1)[:, np.newaxis]
        return res

    # this should work once outer product is implemented
    def _broadcast_iid_future(self, method, **kwargs):
        one_dist_res = getattr(self.distribution, method)(**kwargs)
        if self.ndim == 0:
            return one_dist_res

        # methods that return a single column always
        METHODS_TO_1D = ["energy"]

        if isinstance(one_dist_res, pd.DataFrame):
            one_dist = one_dist_res.values

        if method in METHODS_TO_1D:
            target_shape = (self.shape[0], 1)
        else:
            target_shape = self.shape
        bc_res = np.broadcast_to(one_dist, target_shape)
        return bc_res

    def _iloc(self, rowidx=None, colidx=None):
        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._iat(rowidx, colidx)
        if is_scalar_notnone(rowidx):
            rowidx = pd.Index([rowidx])
        if is_scalar_notnone(colidx):
            colidx = pd.Index([colidx])

        if rowidx is not None:
            new_index = self.index[rowidx]
        else:
            new_index = self.index

        if colidx is not None:
            new_columns = self.columns[colidx]
        else:
            new_columns = self.columns

        if not self._bc_cols:
            distr = self.distribution.iloc[:, colidx]
        else:
            distr = self.distribution

        return IID(distribution=distr, index=new_index, columns=new_columns)

    def _iat(self, rowidx=None, colidx=None):
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")
        self_subset = self.iloc[[rowidx], [colidx]]
        return type(self)(distribution=self_subset.distribution.iat[0, 0])

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        return self._broadcast_iid("mean")

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        return self._broadcast_iid("var")

    def _pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pdf values at the given points
        """
        return self._broadcast_iid("pdf", x=x)

    def _log_pdf(self, x):
        """Logarithmic probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            log pdf values at the given points
        """
        return self._broadcast_iid("log_pdf", x=x)

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
        return self._broadcast_iid("cdf", x=x)

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
        return self._broadcast_iid("ppf", p=p)

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Private method, to be implemented by subclasses.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        return self._broadcast_iid("energy")

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|]`, where :math:`X` is a copy of self,
        and :math:`x` is a constant.

        Private method, to be implemented by subclasses.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to compute energy w.r.t. to

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        return self._broadcast_iid("energy", x=x)

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
        # if delegate has ppf,
        dist_has_impl = self.distribution._has_implementation_of
        if dist_has_impl("_ppf") or dist_has_impl("ppf"):
            return super().sample(n_samples=n_samples)

        # else we sample manually, this will be less efficient due to loops
        if n_samples is None:
            n = 1
        else:
            n = n_samples
        target_shape = (n * self.shape[0], self.shape[1])
        samples = np.zeros(target_shape)

        if self._bc_cols:
            for i, j in np.ndindex(target_shape):
                samples[i, j] = self.distribution.sample()
        else:
            for i in range(target_shape[0]):
                samples[i] = self.distribution.sample()

        if n_samples is None:
            res_index = self.index
        else:
            res_index = pd.MultiIndex.from_product([range(n), self.index])
        return pd.DataFrame(samples, index=res_index, columns=self.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from skpro.distributions import Normal

        n_scalar = Normal(mu=0, sigma=1)
        # array case examples
        params1 = {
            "distribution": n_scalar,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }

        n_row = Normal(mu=[1, 2], sigma=1, columns=pd.Index(["c", "d"]))
        params2 = {
            "distribution": n_row,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),  # this should override n_row.columns
        }

        # scalar case example - corner case, iid does nothing
        params3 = {"distribution": n_scalar}
        return [params1, params2, params3]


def is_scalar_notnone(obj):
    """Check if obj is scalar and not None."""
    return obj is not None and np.isscalar(obj)
