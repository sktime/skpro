"""Bayesian proportion estimator for probabilistic regression."""
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to skpro should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright


__author__ = ["meraldoantonio"]

from skpro.distributions import Beta
from skpro.regression.base import BaseProbaRegressor


class BayesianProportionEstimator(BaseProbaRegressor):
    """Bayesian probabilistic estimator for proportions.

    This estimator uses a Beta prior and Beta posterior with a Binomial likelihood
    for Bayesian inference of proportions. It provides methods for updating the
    posterior, making predictions, and various utilities for analysis and visualization.

    Parameters
    ----------
    prior_alpha : float, optional (default=1)
        Alpha parameter of the Beta prior distribution.
    prior_beta : float, optional (default=1)
        Beta parameter of the Beta prior distribution.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["meraldoantonio"],
        "python_dependencies": ["scipy", "matplotlib"],
        "capability:multioutput": False,
        "capability:missing": True,
        # estimator tags
        # --------------
        "capability:multioutput": False,  # can the estimator handle multi-output data?
        "capability:missing": True,  # can the estimator handle missing data?
        "X_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit, _predict
        "y_inner_mtype": "pd_Series_Table",  # type seen in internal _fit
    }

    def __init__(self, prior_alpha=None, prior_beta=None, prior=None):
        """Initialize the Bayesian inference class with priors.

        Parameters
        ----------
        prior_alpha : float, optional
            The alpha parameter for the Beta prior distribution. Default is None.
        prior_beta : float, optional
            The beta parameter for the Beta prior distribution. Default is None.
        prior : Beta, optional
            An existing Beta distribution prior. Default is None.

        Raises
        ------
        ValueError
            If neither (prior_alpha and prior_beta) nor prior are provided.
        TypeError
            If the provided prior is not an instance of Beta.
        """
        if prior is None:
            if prior_alpha is None or prior_beta is None:
                raise ValueError(
                    "Must provide either (prior_alpha and prior_beta) or prior."
                )
            self.prior_alpha = prior_alpha
            self.prior_beta = prior_beta
            self.prior = Beta(alpha=prior_alpha, beta=prior_beta)
        else:
            if not isinstance(prior, Beta):
                raise TypeError("Prior must be an instance of Beta.")
            self.prior = prior
            self.prior_alpha = prior.alpha
            self.prior_beta = prior.beta

        super().__init__()

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to;
            will be ignored
        y : pandas Series, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        assert y.apply(
            lambda x: isinstance(x, bool) or x in [0, 1]
        ).all(), "Values in y must be boolean or convertible to boolean (0 or 1)"
        self._posterior = self._perform_bayesian_inference(self.prior, X, y)
        return self

    def _predict_proba(self, X=None):
        """Predict distribution over labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for;
            will be ignored

        Returns
        -------
        y_pred : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        y_pred = Beta(alpha=self._posterior.alpha, beta=self._posterior.beta)
        return y_pred

    def _perform_bayesian_inference(self, prior, X, y):
        """Perform Bayesian inference using a conjugate prior (Beta distribution).

        This method calculates the posterior Beta distribution parameters
        given observed binary outcomes.

        Parameters
        ----------
        prior : Beta
            The prior Beta distribution from skpro distributions.
        X : pandas DataFrame
            Feature data corresponding to the observed outcomes `y`.
        y : array-like, must be binary (0 or 1)
            Observed binary outcomes.

        Returns
        -------
        posterior : Beta
            The posterior Beta distribution with updated parameters.
        """
        n = len(y)
        successes = y.sum()
        posterior_alpha = prior.alpha + successes
        posterior_beta = prior.beta + n - successes
        return Beta(alpha=posterior_alpha, beta=posterior_beta)

    def update(self, X, y):
        """Update the posterior with new data.

        Parameters
        ----------
        X : pandas DataFrame
            New feature instances
        y : pandas Series
            New labels

        Returns
        -------
        self : reference to self
        """
        self._posterior = self._perform_bayesian_inference(self._posterior, X, y)
        return self

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
        """
        params1 = {"prior_alpha": 1, "prior_beta": 1}
        params2 = {"prior_alpha": 2, "prior_beta": 2}

        return [params1, params2]
