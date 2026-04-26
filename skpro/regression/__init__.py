"""Probabilitistic supervised regression estimators."""

from skpro.regression.conformal import (
    MapieConformalizedQuantileRegressor,
    MapieCrossConformalRegressor,
    MapieSplitConformalRegressor,
)
from skpro.regression.jackknife import MapieJackknifeAfterBootstrapRegressor
from skpro.regression.nonparametric import NadarayaWatsonCDE
from skpro.regression.shrinking_interval import ShrinkingNormalIntervalRegressor

__all__ = [
    "MapieSplitConformalRegressor",
    "MapieCrossConformalRegressor",
    "MapieConformalizedQuantileRegressor",
    "MapieJackknifeAfterBootstrapRegressor",
    "NadarayaWatsonCDE",
    "ShrinkingNormalIntervalRegressor",
]
