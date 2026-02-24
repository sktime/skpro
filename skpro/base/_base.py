"""Base class and template for regressors and transformers."""
from skbase.base import BaseEstimator as _BaseEstimator
from skbase.base import BaseMetaEstimator as _BaseMetaEstimator
from skbase.base import BaseObject as _BaseObject


class _CommonTags:
    """Mixin for common tag definitions to all estimator base classes."""

    # config common to all estimators
    _config = {}

    _tags = {
        "estimator_type": "estimator",
        "authors": "skpro developers",
        "maintainers": "skpro developers",
        "capabilities:update": False,
    }

    @property
    def name(self):
        """Return the name of the object or estimator."""
        return self.__class__.__name__


class BaseObject(_CommonTags, _BaseObject):
    """Base class for fittable objects."""

    def __init__(self):
        super().__init__()


class BaseEstimator(_CommonTags, _BaseEstimator):
    """Base class for fittable objects."""

    def update(self, X, y=None):
        """Update estimator with new data.

        Parameters
        ----------
        X : pd.DataFrame or 2D np.ndarray
            New training instances to update the model.
        y : pd.Series or 1D np.ndarray, default=None
            New training labels for the update.

        Returns
        -------
        self : Reference to self.
        """
        if not self.is_fitted:
            return self.fit(X, y)

        self._update(X, y)
        return self

    def _update(self, X, y=None):
        """Default strategy for update. 
        
        To be overridden by Bayesian estimators that support incremental updates.
        """
        return self.fit(X, y)


class BaseMetaEstimator(_CommonTags, _BaseMetaEstimator):
    """Base class for fittable composite meta-objects."""
    
    def __init__(self):
        super().__init__()