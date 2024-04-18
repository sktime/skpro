"""Probability distribution objects."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

__all__ = ["BaseDistribution", "_DelegatedDistribution"]

from skpro.distributions.base._base import BaseDistribution
from skpro.distributions.base._delegate import _DelegatedDistribution
