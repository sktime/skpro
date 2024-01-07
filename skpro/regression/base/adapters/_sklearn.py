"""Adapters to sklearn linear regressors with probabilistic components."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from skpro.regression.base._delegate import _DelegatedProbaRegressor


class _DelegateWithFittedParamForwarding(_DelegatedProbaRegressor):
    """Common base class for delegates with attribute forwarding.

    Assumes that delegate has an attribute `estimator_`,
    from which fitted attributes are forwarded to self.
    """

    # attribute for _DelegatedProbaRegressor, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedRegressor docstring
    _delegate_name = "_estimator"
    # _estimator, not estimator_, because we do not want to expose it as
    # fitted params - fitted params are instead forwarded

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        estimator = self._get_delegate()
        estimator.fit(X=X, y=y)

        for attr in self.FITTED_PARAMS_TO_FORWARD:
            setattr(self, attr, getattr(estimator.estimator_, attr))

        return self
