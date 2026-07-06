"""Probabilitistic supervised regression estimators."""

from skpro.regression.conformal import (
    MapieConformalizedQuantileRegressor,
    MapieCrossConformalRegressor,
    MapieSplitConformalRegressor,
)
from skpro.regression.deterministic_reduction import DeterministicReductionRegressor
from skpro.regression.jackknife import MapieJackknifeAfterBootstrapRegressor
from skpro.regression.nonparametric import NadarayaWatsonCDE
from skpro.regression.unconditional_distfit import UnconditionalDistfitRegressor

__all__ = [
    "DeterministicReductionRegressor",
    "MapieSplitConformalRegressor",
    "MapieCrossConformalRegressor",
    "MapieConformalizedQuantileRegressor",
    "MapieJackknifeAfterBootstrapRegressor",
    "NadarayaWatsonCDE",
    "UnconditionalDistfitRegressor",
]
