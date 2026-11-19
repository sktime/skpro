# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base class for concat operation."""

__author__ = ["SaiRevanth25"]

import pandas as pd

from skpro.base import BaseMetaEstimator
from skpro.distributions.base import BaseDistribution


class ConcatDistr(BaseMetaEstimator, BaseDistribution):
    """Concatenate the given distributions along specified axis.

    Parameters
    ----------
    distributions : list
        list of distributions
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along
    ignore_index : bool, default False
        If True, do not use the index values along the concatenation axis.
        The resulting axis will be labeled 0, ..., n - 1.
    """

    # for default get_params/set_params from BaseMetaEstimator
    # named_object_parameters points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    # if the estimator is fittable, _BaseMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in fitted_named_object_parameters
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _tags = {
        # packaging info
        # --------------
        "authors": ["SaiRevanth25", "fkiraly"],
        #
        # estimator tags
        # --------------
        "named_object_parameters": "_distributions",
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
    }

    def __init__(
        self,
        distributions,
        axis=0,
        ignore_index=False,
        index=None,
        columns=None,
    ):
        """Initialize concat with list of distributions and axis for concatenation."""
        self.distributions = distributions
        self.axis = axis
        self.ignore_index = ignore_index
        self.index = index
        self.columns = columns

        self._distributions = self._coerce_to_named_object_tuples(distributions)

        super().__init__(index=index, columns=columns)

    def _concat(self, method, *args, **kwargs):
        """Concatenate the distributions along the specified axis."""
        results = []
        for distr in self.distributions:
            results.append(getattr(distr, method)(*args, **kwargs))

        if self.index is None and self.columns is None:
            return pd.concat(results, axis=self.axis, ignore_index=self.ignore_index)
        else:
            concat_df = pd.concat(results, axis=self.axis, ignore_index=True)

            if self.axis == 0 and self.columns is not None:
                concat_df.columns = self.columns
            elif self.axis == 1 and self.index is not None:
                concat_df.index = self.index

            return concat_df

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        return self._concat("mean")

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        return self._concat("var")

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
        return self._concat("pdf", x=x)

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
        return self._concat("log_pdf", x=x)

    def _pmf(self, x):
        """Probability mass function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pmf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pmf values at the given points
        """
        return self._concat("pmf", x=x)

    def _log_pmf(self, x):
        """Logarithmic probability mass function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pmf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            log pmf values at the given points
        """
        return self._concat("log_pmf", x=x)

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
        return self._concat("cdf", x=x)

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
        return self._concat("ppf", p=p)

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Private method, to be implemented by subclasses.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        return self._concat("energy_self")

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
        return self._concat("energy_x", x=x)

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
        if n_samples is None:
            return self._concat("sample")
        else:
            # we concat and sort
            samples = self._concat("sample", n_samples=n_samples)
            samples = samples.sort_index()
            return samples

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from skpro.distributions import Normal, Uniform

        n32 = Normal(0, 1, index=[0, 1, 2], columns=["a", "b"])
        u32 = Uniform(-1, 1, index=[3, 4, 5], columns=["a", "b"])

        params0 = {
            "distributions": [n32, n32],
            "axis": 1,
            "ignore_index": True,
        }
        params1 = {
            "distributions": [n32, u32],
            "ignore_index": False,
        }
        params2 = {
            "distributions": [n32, u32],
            "axis": 1,
            "index": [0, 2, 4],
            "columns": ["a", "b", "foo", "bar"],
        }

        return [params0, params1, params2]
