"""Adapters for probability distribution objects, statsmodels facing."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.distributions.adapters.statsmodels._empirical import empirical_from_rvdf

__all__ = ["empirical_from_rvdf"]
