"""Probabilistic classifiers."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.classification.base import BaseProbaClassifier
from skpro.classification.adapters.sklearn import SklearnClassifierAdapter

__all__ = ["BaseProbaClassifier", "SklearnClassifierAdapter"]
