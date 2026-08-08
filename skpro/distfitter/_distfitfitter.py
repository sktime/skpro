"""Distribution fitter wrapping the distfit package."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.distfitter.base import BaseDistFitter

__author__ = ["areychana"]


class DistfitFitter(BaseDistFitter):
    """Fit a parametric distribution to data using the ``distfit`` package.

    Wraps `distfit <https://github.com/erdogant/distfit>`_'s parametric fitting
    procedure: a set of candidate ``scipy.stats`` distributions is fitted to the
    data, scored by a goodness-of-fit statistic, and the best-scoring
    distribution is returned as a fitted ``skpro`` scalar distribution.

    Only ``distfit``'s ``method="parametric"`` mode is currently supported.
    ``distfit``'s ``"quantile"``, ``"percentile"``, and ``"discrete"`` modes
    produce differently shaped model output that this fitter does not (yet)
    convert into a ``skpro`` distribution.

    Parameters
    ----------
    distr : str or list of str, optional (default="popular")
        Candidate distribution(s) to fit and compare, passed to ``distfit``.
        ``"popular"`` tests ``[norm, expon, pareto, dweibull, t, genextreme,
        gamma, lognorm, beta, uniform, loggamma]``; ``"full"`` tests all
        ``scipy.stats`` continuous distributions; a single name (e.g.
        ``"norm"``) or a list of names restricts to those distributions.
    stats : str, optional (default="RSS")
        Goodness-of-fit statistic used by ``distfit`` to rank candidates.
        One of ``"RSS"``, ``"wasserstein"``, ``"ks"``, ``"energy"``,
        ``"goodness_of_fit"``.
    bins : int or "auto", optional (default="auto")
        Histogram bin size used internally by ``distfit``.
    random_state : int, optional (default=None)
        Random state passed to ``distfit``.

    Attributes
    ----------
    dist_name_ : str
        Name of the best-fitting ``scipy.stats`` distribution, as chosen by
        ``distfit``.
    shape_args_ : tuple
        Fitted shape parameters for ``dist_name_``, as returned by ``distfit``.
    dist_loc_ : float
        Fitted location parameter for ``dist_name_``.
    dist_scale_ : float
        Fitted scale parameter for ``dist_name_``.
    fit_summary_ : pandas.DataFrame
        Full ``distfit`` summary table of all candidate distributions tried,
        with their scores.

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.distfitter import DistfitFitter
    >>> X = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0, 4.5, 3.5, 2.5, 1.5])

    >>> fitter = DistfitFitter(distr="norm")
    >>> fitter.fit(X)
    DistfitFitter(distr='norm')
    >>> dist = fitter.proba()
    """

    _tags = {
        "authors": ["areychana"],
        "python_dependencies": ["distfit"],
    }

    def __init__(self, distr="popular", stats="RSS", bins="auto", random_state=None):
        self.distr = distr
        self.stats = stats
        self.bins = bins
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, C=None):
        """Fit the best-scoring distfit distribution to the data.

        Parameters
        ----------
        X : pandas DataFrame
            Data to fit the distribution to.
        C : ignored

        Returns
        -------
        self : reference to self
        """
        from distfit import distfit as _distfit

        vals = X.values.ravel()

        dfit = _distfit(
            method="parametric",
            distr=self.distr,
            stats=self.stats,
            bins=self.bins,
            random_state=self.random_state,
            verbose="warning",
        )
        dfit.fit_transform(vals)

        model = dfit.model
        self.dist_name_ = model["name"]
        self.shape_args_ = tuple(model["arg"])
        self.dist_loc_ = float(model["loc"])
        self.dist_scale_ = float(model["scale"])
        self.fit_summary_ = dfit.summary

        return self

    def _proba(self):
        """Return the best-fitting distribution found by distfit.

        Returns
        -------
        dist : skpro BaseDistribution (scalar)
        """
        from skpro.distfitter._distfit_adapter import _DistfitDistribution

        return _DistfitDistribution(
            dist_name=self.dist_name_,
            shape_args=self.shape_args_,
            dist_loc=self.dist_loc_,
            dist_scale=self.dist_scale_,
        )

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
        params1 = {"distr": "norm"}
        params2 = {"distr": ["norm", "expon"], "stats": "wasserstein"}
        return [params1, params2]
