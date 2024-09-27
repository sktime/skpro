"""Meta-algorithms to build online regression models."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.regression.online._dont_refit import OnlineDontRefit
from skpro.regression.online._refit import OnlineRefit

__all__ = ["OnlineDontRefit", "OnlineRefit"]
