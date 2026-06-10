"""Coerce foreign regressors to skpro online regressor adapters."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["patelchaitany"]

from skpro.regression.adapters.river._utils import is_river_estimator
from skpro.regression.base import BaseOnlineRegressor


def coerce_to_skpro_regressor(estimator):
    """Wrap foreign online regressors for use in skpro meta-estimators.

    Parameters
    ----------
    estimator : object
        Estimator passed by the user to a meta-estimator such as
        ``BaggingRegressor``.

    Returns
    -------
    estimator : skpro regressor
        ``estimator`` unchanged if already an skpro regressor; otherwise a
        suitable adapter instance.
    """
    if isinstance(estimator, BaseOnlineRegressor):
        return estimator

    if is_river_estimator(estimator):
        from skpro.regression.adapters.river import RiverRegressor

        return RiverRegressor(estimator)

    return estimator
