# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Hyperactive Search CV tuning for probabilistic regressors."""

from skpro.registry._placeholder_rec import _placeholder_record
from skpro.regression.base._delegate import _DelegatedProbaRegressor

@_placeholder_record(
    dependency="hyperactive",
    import_path="hyperactive.integrations.skpro.ProbaRegOptCV"
)
class ProbaRegOptCV(_DelegatedProbaRegressor):
    """Hyperparameter search cross-validation using Hyperactive tuner.

    Performs hyperparameter optimization of probabilistic regressors
    using the hyperactive optimization backend.
    """

    _tags = {
        "estimator_type": "regressor",
        "capability:multioutput": True,
        "capability:missing": True,
        "python_dependencies": "hyperactive",
        "tests:vm": True,
    }

    def __init__(
        self,
        estimator,
        optimizer,
        cv=None,
        scoring=None,
        refit=True,
        error_score=None,
        backend=None,
        backend_params=None,
    ):
        self.estimator = estimator
        self.optimizer = optimizer
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.backend = backend
        self.backend_params = backend_params

        super().__init__()

        # Clone tags from base estimator
        tags_to_clone = [
            "capability:multioutput",
            "capability:missing",
            "capability:survival",
        ]
        self.clone_tags(estimator, tags_to_clone)

    def _fit(self, X, y, C=None):
        """Fit stub placeholder."""
        pass
