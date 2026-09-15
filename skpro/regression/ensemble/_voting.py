"""Voting ensemble of heterogeneous probabilistic regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Ashish-Kumar-Dash"]
__all__ = ["VotingProbaRegressor"]

from skpro.base import BaseMetaEstimator
from skpro.distributions.mixture import Mixture
from skpro.regression.base import BaseProbaRegressor


class VotingProbaRegressor(BaseMetaEstimator, BaseProbaRegressor):
    """Voting ensemble of heterogeneous probabilistic regressors.

    Fits multiple probabilistic regressors on the same training data.
    On ``predict_proba``, returns a ``Mixture`` distribution of the
    component predictions, with user-specified or uniform weights.

    Generalizes ``sklearn``'s ``VotingRegressor`` to the probabilistic
    regression setting, where predictions are full distributions
    rather than point predictions.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples or list of estimators
        The ensemble members. Each estimator must be a descendant of
        ``BaseProbaRegressor``.
        If a plain list of estimators is passed, names are generated
        automatically from class names.
    weights : array-like of float, optional, default=None
        Mixture weights for the component predictions.
        If None, uniform weights are used.
        Weights are normalized to sum to 1 internally by ``Mixture``.

    Attributes
    ----------
    estimators_ : list of (str, estimator) tuples
        Fitted clones of the estimators passed in ``estimators``.

    Examples
    --------
    >>> from skpro.regression.ensemble import VotingProbaRegressor
    >>> from skpro.regression.residual import ResidualDouble
    >>> from skpro.regression.linear import BayesianRidge
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg1 = ResidualDouble(LinearRegression())
    >>> reg2 = BayesianRidge()
    >>>
    >>> voter = VotingProbaRegressor(
    ...     estimators=[("r1", reg1), ("r2", reg2)],
    ... )
    >>> voter.fit(X_train, y_train)
    VotingProbaRegressor(...)
    >>> y_pred = voter.predict_proba(X_test)
    """

    _tags = {
        "object_type": "regressor_proba",
        "estimator_type": "regressor_proba",
        "named_object_parameters": "_estimators",
        "fitted_named_object_parameters": "estimators_",
        "capability:missing": True,
        "capability:survival": True,
    }

    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        est_list = self._estimators

        # self can handle missing data if and only if all components can
        self._anytagis_then_set("capability:missing", False, True, est_list)

        # self can handle survival data if and only if at least one component can
        self._anytagis_then_set("capability:survival", True, False, est_list)

    @property
    def _estimators(self):
        return self._coerce_to_named_object_tuples(self.estimators, clone=False)

    @_estimators.setter
    def _estimators(self, value):
        self.estimators = value

    def _fit(self, X, y, C=None):
        """Fit all component regressors to training data.

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to
        C : pandas DataFrame, optional (default=None)
            censoring information for survival analysis

        Returns
        -------
        self : reference to self
        """
        self.estimators_ = []

        for name, est in self._estimators:
            fitted_est = est.clone().fit(X, y, C=C)
            self.estimators_.append((name, fitted_est))

        return self

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        Returns a ``Mixture`` of predictions from all fitted regressors.

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in ``fit``
            data to predict labels for

        Returns
        -------
        y_pred : skpro ``Mixture`` distribution, same length as ``X``
            mixture of probabilistic predictions from all component regressors
        """
        y_probas = [(name, est.predict_proba(X)) for name, est in self.estimators_]

        return Mixture(distributions=y_probas, weights=self.weights)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sklearn.linear_model import LinearRegression

        from skpro.regression.linear import BayesianRidge
        from skpro.regression.residual import ResidualDouble

        reg1 = ResidualDouble(LinearRegression())
        reg2 = BayesianRidge()

        params1 = {
            "estimators": [("r1", reg1), ("r2", reg2)],
        }
        params2 = {
            "estimators": [("r1", reg1), ("r2", reg2)],
            "weights": [0.7, 0.3],
        }
        params3 = {
            "estimators": [reg1, reg2],
        }

        return [params1, params2, params3]
