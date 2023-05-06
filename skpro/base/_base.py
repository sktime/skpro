# -*- coding: utf-8 -*-
"""Base class and template for regressors and transformers."""
from skbase.base import (
    BaseEstimator as _BaseEstimator,
    BaseMetaEstimator as _BaseMetaEstimator,
)


class _CommonTags():
    """Mixin for common tag definitions to all estimator base classes."""

    # config common to all estimators
    _config = {}

    _tags = {"estimator_type": "estimator"}


class BaseEstimator(_CommonTags, _BaseEstimator):
    """Base class for fittable objects."""


class BaseMetaEstimator(_CommonTags, _BaseMetaEstimator):
    """Base class for fittable composite meta-objects."""
