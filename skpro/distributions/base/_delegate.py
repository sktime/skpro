"""Delegator mixin that delegates all methods to wrapped distribution.

Useful for building estimators where all but one or a few methods are delegated. For
that purpose, inherit from this estimator and then override only the methods that
are not delegated.
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["_DelegatedDistribution"]

from copy import deepcopy

from skpro.distributions.base._base import BaseDistribution


class _DelegatedDistribution(BaseDistribution):
    """Delegator mixin that delegates all methods to wrapped estimator.

    Delegates inner methods to a wrapped estimator.
        Wrapped estimator is value of attribute with name self._delegate_name.
        By default, this is "estimator_", i.e., delegates to self.estimator_
        To override delegation, override _delegate_name attribute in child class.

    Delegates the following methods:
        _iloc, pdf, log_pdf, cdf, ppf, energy, mean, var, pdfnorm, sample

    Does NOT delegate get_params, set_params.
        get_params, set_params will hence use one additional nesting level by default.

    Does NOT delegate or copy tags, this should be done in a child class if required.
    """

    # attribute for _DelegatedDistribution, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedDistribution docstring
    _delegate_name = "delegate_"

    def _get_delegate(self):
        return getattr(self, self._delegate_name)

    def _iloc(self, rowidx=None, colidx=None):
        cls = self.__class__

        delegate = self._get_delegate()
        delegate_subset = delegate.iloc[rowidx, colidx]
        delegate_subset_params = deepcopy(delegate_subset.get_params())

        return cls(**delegate_subset_params)

    def pdf(self, x):
        r"""Probability density function.

        Let :math:`X` be a random variables with the distribution of `self`,
        taking values in `(N, n)` `DataFrame`-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`p_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input `x` representing :math:`x`,
        is a `DataFrame` with same columns and indices as `self`,
        and entries :math:`p_{X_{ij}}(x_{ij})`.

        Parameters
        ----------
        x : `pandas.DataFrame` or 2D np.ndarray
            representing :math:`x`, as above

        Returns
        -------
        `DataFrame` with same columns and index as `self`
            containing :math:`p_{X_{ij}}(x_{ij})`, as above
        """
        delegate = self._get_delegate()
        return delegate.pdf(x)

    def log_pdf(self, x):
        r"""Logarithmic probability density function.

        Numerically more stable than calling pdf and then taking logartihms.

        Let :math:`X` be a random variables with the distribution of `self`,
        taking values in `(N, n)` `DataFrame`-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`p_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input `x` representing :math:`x`,
        is a `DataFrame` with same columns and indices as `self`,
        and entries :math:`\log p_{X_{ij}}(x_{ij})`.

        If `self` has a mixed or discrete distribution, this returns
        the weighted continuous part of `self`'s distribution instead of the pdf,
        i.e., the marginal pdf integrate to the weight of the continuous part.

        Parameters
        ----------
        x : `pandas.DataFrame` or 2D np.ndarray
            representing :math:`x`, as above

        Returns
        -------
        `DataFrame` with same columns and index as `self`
            containing :math:`\log p_{X_{ij}}(x_{ij})`, as above
        """
        delegate = self._get_delegate()
        return delegate.log_pdf(x)

    def cdf(self, x):
        """Cumulative distribution function."""
        delegate = self._get_delegate()
        return delegate.cdf(x)

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        delegate = self._get_delegate()
        return delegate.ppf(p)

    def energy(self, x=None):
        r"""Energy of self, w.r.t. self or a constant frame x.

        Let :math:`X, Y` be i.i.d. random variables with the distribution of `self`.

        If `x` is `None`, returns :math:`\mathbb{E}[|X-Y|]` (for each row),
        "self-energy" (of the row marginal distribution).
        If `x` is passed, returns :math:`\mathbb{E}[|X-x|]` (for each row),
        "energy wrt x" (of the row marginal distribution).

        Parameters
        ----------
        x : None or pd.DataFrame, optional, default=None
            if pd.DataFrame, must have same rows and columns as `self`

        Returns
        -------
        pd.DataFrame with same rows as `self`, single column `"energy"`
        each row contains one float, self-energy/energy as described above.
        """
        delegate = self._get_delegate()
        return delegate.energy(x=x)

    def mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        delegate = self._get_delegate()
        return delegate.mean()

    def var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns :math:`\mathbb{V}[X] = \mathbb{E}\left(X - \mathbb{E}[X]\right)^2`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        delegate = self._get_delegate()
        return delegate.var()

    def pdfnorm(self, a=2):
        r"""a-norm of pdf, defaults to 2-norm.

        computes a-norm of the entry marginal pdf, i.e.,
        :math:`\mathbb{E}[p_X(X)^{a-1}] = \int p(x)^a dx`,
        where :math:`X` is a random variable distributed according to the entry marginal
        of `self`, and :math:`p_X` is its pdf

        Parameters
        ----------
        a: int or float, optional, default=2

        Returns
        -------
        pd.DataFrame with same rows and columns as `self`
        each entry is :math:`\mathbb{E}[p_X(X)^{a-1}] = \int p(x)^a dx`, see above
        """
        delegate = self._get_delegate()
        return delegate.pdfnorm(a=a)

    def sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None

        Returns
        -------
        if `n_samples` is `None`:
        returns a sample that contains a single sample from `self`,
        in `pd.DataFrame` mtype format convention, with `index` and `columns` as `self`
        if n_samples is `int`:
        returns a `pd.DataFrame` that contains `n_samples` i.i.d. samples from `self`,
        in `pd-multiindex` mtype format convention, with same `columns` as `self`,
        and `MultiIndex` that is product of `RangeIndex(n_samples)` and `self.index`
        """
        delegate = self._get_delegate()
        return delegate.sample(n_samples=n_samples)
