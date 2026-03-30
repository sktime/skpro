"""Probabilitistic supervised regression estimators."""

from skpro.regression.conformal import (
    MapieConformalizedQuantileRegressor,
    MapieCrossConformalRegressor,
    MapieSplitConformalRegressor,
)
from skpro.regression.jackknife import MapieJackknifeAfterBootstrapRegressor
from skpro.regression.nonparametric import NadarayaWatsonCDE
from skpro.regression.reducing_interval import ReducingIntervalRegressor

__all__ = [
    "MapieSplitConformalRegressor",
    "MapieCrossConformalRegressor",
    "MapieConformalizedQuantileRegressor",
    "MapieJackknifeAfterBootstrapRegressor",
    "NadarayaWatsonCDE",
    "ReducingIntervalRegressor",
]
