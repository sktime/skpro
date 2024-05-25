"""Base classes for probabilistic regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["BaseProbaRegressor", "_DelegatedProbaRegressor"]

from skpro.regression.base._base import BaseProbaRegressor
from skpro.regression.base._delegate import _DelegatedProbaRegressor
