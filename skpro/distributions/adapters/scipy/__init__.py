"""Adapters for probability distribution objects, scipy facing."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.distributions.adapters.scipy._empirical import empirical_from_discrete
from skpro.distributions.adapters.scipy._distribution import _ScipyAdapter

__all__ = ["empirical_from_discrete", "_ScipyAdapter"]
