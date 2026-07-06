"""Probabilitistic supervised regression estimators."""

from skpro.regression.bayesian_proportion import BayesianProportionEstimator
from skpro.regression.conformal import (
    MapieConformalizedQuantileRegressor,
    MapieCrossConformalRegressor,
    MapieSplitConformalRegressor,
)
from skpro.regression.jackknife import MapieJackknifeAfterBootstrapRegressor
from skpro.regression.nonparametric import NadarayaWatsonCDE

__all__ = [
    "BayesianProportionEstimator",
    "MapieSplitConformalRegressor",
    "MapieCrossConformalRegressor",
    "MapieConformalizedQuantileRegressor",
    "MapieJackknifeAfterBootstrapRegressor",
    "NadarayaWatsonCDE",
]
