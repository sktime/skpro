"""Outlier detection based on probabilistic regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__all__ = [
    "QuantileOutlierDetector",
    "DensityOutlierDetector",
    "LossOutlierDetector",
]

from skpro.outlier._density import DensityOutlierDetector
from skpro.outlier._loss import LossOutlierDetector
from skpro.outlier._quantile import QuantileOutlierDetector
