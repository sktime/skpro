"""Base classes for adapting probabilistic regressors to the skproframework."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["_DelegateWithFittedParamForwarding"]

from skpro.regression.base.adapters._sklearn import _DelegateWithFittedParamForwarding
