"""Adapters for probabilistic regressors, towards sklearn."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.regression.adapters.ngboost._ngboost_proba import NGBoostAdapter

__all__ = ["NGBoostAdapter"]
