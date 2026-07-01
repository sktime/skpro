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
        self.__post_init__()

    def __dynamic_tags__(self):
        """Set dynamic tags conditional on parameters.

        Override this method to set tags that depend on the estimator's
        parameters, e.g., cloning tags from a component estimator.
        This is called at the end of ``__init__``, after ``super().__init__()``.
        """

    def __post_init__(self):
        """Initialize non-parameter attributes and validate parameters.

        Override this method to place parameter checks or initialization
        of derived quantities that should not be constructor parameters.
        This is called at the end of ``__init__``, after ``__dynamic_tags__``.

        Avoid overriding ``__init__`` directly; place any such logic here.
        """


class BaseEstimator(_CommonTags, _BaseEstimator):
    """Base class for fittable objects."""

    def __init__(self):
        super().__init__()
        self.__dynamic_tags__()
        self.__post_init__()

    def __dynamic_tags__(self):
        """Set dynamic tags conditional on parameters.

        Override this method to set tags that depend on the estimator's
        parameters, e.g., cloning tags from a component estimator.
        This is called at the end of ``__init__``, after ``super().__init__()``.
        """

    def __post_init__(self):
        """Initialize non-parameter attributes and validate parameters.

        Override this method to place parameter checks or initialization
        of derived quantities that should not be constructor parameters.
        This is called at the end of ``__init__``, after ``__dynamic_tags__``.

        Avoid overriding ``__init__`` directly; place any such logic here.
        """


class BaseMetaEstimator(_CommonTags, _BaseMetaEstimator):
    """Base class for fittable composite meta-objects."""
