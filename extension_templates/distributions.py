"""Extension template for probability distributions - simple pattern."""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to skpro should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

from skpro.distributions.base import BaseDistribution

# todo: add any necessary imports here - no soft dependency imports

# todo: for imports of skpro soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class ClassName(BaseDistribution):
    """Custom probability distribution. todo: write docstring.

    todo: describe your custom probability distribution here

    Parameters
    ----------
    parama : float or np.ndarray
        descriptive explanation of parama
    paramb : float or np.ndarray, optional (default='default')
        descriptive explanation of paramb
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    # tags inherited from base are "safe defaults" which can usually be left as-is
    _tags = {
        # packaging info
        # --------------
        "authors": ["author1", "author2"],  # authors, GitHub handles
        "maintainers": ["maintainer1", "maintainer2"],  # maintainers, GitHub handles
        # author = significant contribution to code at some point
        # maintainer = algorithm maintainer role, "owner"
        # specify one or multiple authors and maintainers, only for skpro contribution
        # remove maintainer tag if maintained by skpro/sktim core team
        #
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # PEP 440 python dependencies specifier,
        # e.g., "numba>0.53", or a list, e.g., ["numba>0.53", "numpy>=1.19.0"]
        # delete if no python dependencies or version limitations
        #
        # estimator tags
        # --------------
        "distr:measuretype": "continuous",  # one of "discrete", "continuous", "mixed"
        # these tags should correspond to which methods are numerically exact
        # and which are approximations, e.g., using Monte Carlo
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        # leave the broadcast_init tag as-is, this tag exists for compatibility with
        # distributions deviating from assumptions on input parameters, e.g., Empirical
        "broadcast_init": "on",
    }

    # todo: fill init
    # params should be written to self and never changed
    # super call must not be removed, change class name
    # parameter checks can go after super call
    def __init__(self, param1, param2="param2default", index=None, columns=None):
        # all distributions must have index and columns arg with None defaults
        # this is to ensure pandas-like behaviour

        # todo: write any hyper-parameters and components to self
        self.param1 = param1
        self.param2 = param2

        # leave this as is
        super().__init__(index=index, columns=columns)

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

    # todo: implement as many of the following methods as possible
    # if not implemented, the base class will try to fill it in
    # from the other implemented methods
    # at least _ppf, or sample should be implemented for the distribution to be usable
    # if _ppf is implemented, sample does not need to be implemented (uses ppf sampling)

    # todo: consider implementing
    # if not implemented, uses Monte Carlo estimate via sample
    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
        param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

        res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
        return res

    # todo: consider implementing
    # if not implemented, uses Monte Carlo estimate via sample
    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
        param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

        res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
        return res

    # todo: consider implementing - only for continuous or mixed distributions
    # at least one of _pdf and _log_pdf should be implemented
    # if not implemented, returns exp of log_pdf
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
        param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
        param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

        res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
        return res

    # todo: consider implementing - only for continuous or mixed distributions
    # at least one of _pdf and _log_pdf should be implemented
    # if not implemented, returns log of pdf
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
        param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
        param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

        res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
        return res

    # todo: consider implementing - only for discrete or mixed distributions
    # at least one of _pmf and _log_pmf should be implemented
    # if not implemented, returns exp of log_pmf
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
        param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
        param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

        res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
        return res

    # todo: consider implementing - only for discrete or mixed distributions
    # at least one of _pmf and _log_pmf should be implemented
    # if not implemented, returns log of pmf
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
        param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
        param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

        res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
        return res

    # todo: consider implementing
    # at least one of _ppf and sample must be implemented
    # if not implemented, uses Monte Carlo estimate based on sample
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
        param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
        param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

        res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
        return res

    # todo: consider implementing
    # at least one of _ppf and sample must be implemented
    # if not implemented, uses bisection method on cdf
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
        pass

    # todo: consider implementing
    # if not implemented, uses Monte Carlo estimate via sample
    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Private method, to be implemented by subclasses.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
        param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

        res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
        return res

    # todo: consider implementing
    # if not implemented, uses Monte Carlo estimate via sample
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
        param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
        param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

        res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
        return res

    # todo: consider implementing
    # at least one of _ppf and sample must be implemented
    # if not implemented, uses _ppf for sampling (inverse cdf on uniform)
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
        param1 = self._bc_params["param1"]  # returns broadcast params to x.shape
        param2 = self._bc_params["param2"]  # returns broadcast params to x.shape

        res = "do_sth_with(" + param1 + param2 + ")"  # replace this by internal logic
        return res

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
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

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        #
        # this can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from skpro or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params
        # params = {"est": value3, "parama": value4}
        # return params
