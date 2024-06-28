"""Adapters for probabilistic classifiers, towards sklearn."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.classification.sklearn_classifiers._sklearn_proba_class import (
    MulticlassSklearnProbaClassifier,
)

__all__ = ["MulticlassSklearnProbaClassifier"]
