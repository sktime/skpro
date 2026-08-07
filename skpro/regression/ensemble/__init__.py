"""Ensemble probabilistic regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.regression.ensemble._bagging import BaggingRegressor
from skpro.regression.ensemble._ngboost import NGBoostRegressor
from skpro.regression.ensemble._stacking import StackingProbaRegressor
from skpro.regression.ensemble._voting import VotingProbaRegressor

__all__ = [
    "BaggingRegressor",
    "NGBoostRegressor",
    "StackingProbaRegressor",
    "VotingProbaRegressor",
]
