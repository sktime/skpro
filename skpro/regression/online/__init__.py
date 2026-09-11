"""Meta-algorithms to build online regression models."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.regression.online._batch_mixture import OnlineBatchMixture
from skpro.regression.online._dont_refit import OnlineDontRefit
from skpro.regression.online._exponential_forgetting import OnlineExponentialForgetting
from skpro.regression.online._refit import OnlineRefit
from skpro.regression.online._refit_every import OnlineRefitEveryN
from skpro.regression.online._sliding_window import OnlineSlidingWindow

__all__ = [
    "OnlineBatchMixture",
    "OnlineDontRefit",
    "OnlineExponentialForgetting",
    "OnlineRefit",
    "OnlineRefitEveryN",
    "OnlineSlidingWindow",
]
