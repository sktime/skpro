# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Composition for mean/scale family of distributions."""

__author__ = ["fkiraly"]

import numpy as np

from skpro.distributions.base import BaseDistribution


class MeanScale(BaseDistribution):
    r"""Composition for mean/scale family of distributions.

    This distribution parameterizes the natural mean/scale family of distirbutions
    generated by the component distribution ``d``,
    parameterized a mean offset :math:`\mu`, represented by ``mu``, and
    scale factor :math:`s`, represented by ``scale``

    That is, if :math:`X` is distributed according to ``d``, then
    this distribution represents for parameters ``d``, ``mu``, ``scale``,
    the distribution of the random variable :math:`\mu + s \cdot X`.

    For instance, the cdf of this distribution is

    .. math:: F(x) = F_X\left(\frac{1}{s}(x - \mu)\right)

    Parameters
    ----------
    d : BaseDistribution, skpro distribution instance
        component distribution
    mu : float or array of float (1D or 2D)
        mean of the normal distribution
    sigma : float or array of float (1D or 2D), must be positive
        standard deviation of the normal distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.normal import Normal

    >>> n = Normal(mu=[[0, 1], [2, 3], [4, 5]])
    >>> d = MeanScale(d=n, mu=2, sigma=3)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
        "broadcast_params": ["mu", "sigma"],
    }

    def __init__(self, d, mu=0, sigma=1, index=None, columns=None):
        self.d = d
        self.mu = mu
        self.sigma = sigma

        if d.index is not None and index is None:
            index = d.index
        if d.columns is not None and columns is None:
            columns = d.columns

        super().__init__(index=index, columns=columns)

    # def _iloc(self, rowidx=None, colidx=None):
    #     dist_subset = self.d.iloc[rowidx, colidx]

    #     return MeanScale(d=dist_subset, mu=self.mu, sigma=self.sigma)

    # def _iat(self, rowidx=None, colidx=None):
    #     dist_subset = self.d.iat[rowidx, colidx]
    #     return MeanScale(d=dist_subset, mu=self.mu, sigma=self.sigma)

    def _subset_params(self, rowidx, colidx, coerce_scalar=False):
        """Subset distribution parameters to given rows and columns.

        Parameters
        ----------
        rowidx : None, numpy index/slice coercible, or int
            Rows to subset to. If None, no subsetting is done.
        colidx : None, numpy index/slice coercible, or int
            Columns to subset to. If None, no subsetting is done.
        coerce_scalar : bool, optional, default=False
            If True, and the subsetted parameter is a scalar, coerce it to a scalar.

        Returns
        -------
        dict
            Dictionary with subsetted distribution parameters.
            Keys are parameter names of ``self``, values are the subsetted parameters.
        """
        params = self.get_params()
        subset_params = ["mu", "sigma"]

        subset_param_dict = {}
        for param in subset_params:
            val = params[param]
            subset_param_dict[param] = self._subset_param(
                val=val,
                rowidx=rowidx,
                colidx=colidx,
                coerce_scalar=coerce_scalar,
            )
        if coerce_scalar:
            subset_param_dict["d"] = self.d.iat[rowidx, colidx]
        else:
            subset_param_dict["d"] = self.d.iloc[rowidx, colidx]
        return subset_param_dict

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        return self._bc_params["mu"] + self.d.mean()

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        return self._bc_params["sigma"] ** 2 * self.d.var()

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
        mu = self._bc_params["mu"]
        scale = self._bc_params["sigma"]

        x_inner = (x - mu) / scale
        pdf_arr = self.d.pdf(x_inner) / scale
        return pdf_arr

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
        mu = self._bc_params["mu"]
        scale = self._bc_params["sigma"]

        x_inner = (x - mu) / scale
        lpdf_arr = self.d.log_pdf(x_inner) - np.log(scale)
        return lpdf_arr

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
        mu = self._bc_params["mu"]
        scale = self._bc_params["sigma"]

        x_inner = (x - mu) / scale
        cdf_arr = self.d.cdf(x_inner)
        return cdf_arr

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
        mu = self._bc_params["mu"]
        scale = self._bc_params["sigma"]

        icdf_arr = mu + scale * self.d.ppf(p)
        return icdf_arr

        # def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Private method, to be implemented by subclasses.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        # scale = self._bc_params["sigma"]

        # en_arr = scale * self.d.energy()
        # return en_arr

        # def _energy_x(self, x):
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
        # mu = self._bc_params["mu"]
        # scale = self._bc_params["sigma"]

        # x_offset = (x - mu) / scale
        # en_arr = scale * self.d.energy(x_offset)
        # return en_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from skpro.distributions.normal import Normal

        d = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=2)
        dsc = Normal(0, 1)

        # array case examples
        params0 = {"d": d}
        params1 = {"d": d, "mu": 1}
        params2 = {"d": d, "sigma": 2}
        params3 = {"d": d, "mu": 2, "sigma": 3}
        params4 = {"d": d, "mu": [[0, 1], [2, 3], [4, 5]], "sigma": 2}
        # params5 = {
        #     "d": dsc,
        #     "mu": 0,
        #     "sigma": 1,
        #     "index": pd.Index([1, 2, 5]),
        #     "columns": pd.Index(["a", "b"]),
        # }

        # scalar case examples
        params6 = {"d": dsc}
        params7 = {"d": dsc, "mu": 1, "sigma": 2}
        return [
            params0,
            params1,
            params2,
            params3,
            params4,
            # params5,
            params6,
            params7,
        ]
