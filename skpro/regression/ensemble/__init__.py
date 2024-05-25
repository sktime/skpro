"""Natural Gradient Boosting Regressor models."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.regression.ensemble._bagging import BaggingRegressor
from skpro.regression.ensemble._ngboost import NGBoostRegressor

__all__ = ["BaggingRegressor", "NGBoostRegressor"]
