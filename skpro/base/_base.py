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
    }

    @property
    def name(self):
        """Return the name of the object or estimator."""
        return self.__class__.__name__


class BaseObject(_CommonTags, _BaseObject):
    """Base class for fittable objects."""

    def __init__(self):
        super().__init__()
        self.__dynamic_tags__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        pass

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor

        IMPORTANT: no significant compute or memory use should happen in __post_init__,
        memory and compute intensive operations should be in _fit, not __post_init__.
        """
        pass


class BaseEstimator(_CommonTags, _BaseEstimator):
    """Base class for fittable objects."""

    def __init__(self):
        super().__init__()
        self.__dynamic_tags__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        pass

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor

        IMPORTANT: no significant compute or memory use should happen in __post_init__,
        memory and compute intensive operations should be in _fit, not __post_init__.
        """
        pass


class BaseMetaEstimator(_CommonTags, _BaseMetaEstimator):
    """Base class for fittable composite meta-objects."""
