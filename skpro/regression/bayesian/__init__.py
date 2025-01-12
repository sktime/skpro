"""Base classes for Bayesian probabilistic regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__all__ = [
    "BayesianConjugateLinearRegressor",
    "BayesianLinearRegressor",
]

from skpro.regression.bayesian._linear_conjugate import BayesianConjugateLinearRegressor
from skpro.regression.bayesian._linear_mcmc import BayesianLinearRegressor
