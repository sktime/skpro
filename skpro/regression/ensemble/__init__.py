"""Natural Gradient Boosting Regressor models."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.regression.ensemble._bagging import BaggingRegressor
from skpro.regression.ensemble._ngboost import NGBoostRegressor
from skpro.regression.ensemble._probabilistic_ensemble import (
    ProbabilisticBoostingRegressor,
    ProbabilisticStackingRegressor,
)

__all__ = [
    "BaggingRegressor",
    "NGBoostRegressor",
    "ProbabilisticStackingRegressor",
    "ProbabilisticBoostingRegressor",
]
