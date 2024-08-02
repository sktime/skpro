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

        import sys
        from warnings import warn

        from packaging.specifiers import SpecifierSet

        py39_or_higher = SpecifierSet(">=3.9")
        sys_version = sys.version.split(" ")[0]

        # todo 2.6.0 - check whether python 3.8 eol is reached.
        # If yes, remove this msg.
        if sys_version not in py39_or_higher:
            warn(
                f"From skpro 2.5.0, skpro requires Python version >=3.9, "
                f"but found {sys_version}. "
                "The package can still be installed, until 3.8 end of life "
                "is reached, "
                "but some functionality may not work as test coverage is dropped."
                "Kindly note for context: python 3.8 will reach end of life "
                "in October 2024, and multiple skpro core dependencies, "
                "including scikit-learn, have already dropped support for 3.8. ",
                category=DeprecationWarning,
                stacklevel=2,
            )


class BaseEstimator(_CommonTags, _BaseEstimator):
    """Base class for fittable objects."""


class BaseMetaEstimator(_CommonTags, _BaseMetaEstimator):
    """Base class for fittable composite meta-objects."""
