"""Clone plugin for River estimators."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["patelchaitany"]

from copy import deepcopy

from skbase.base._clone_plugins import BaseCloner

from skpro.regression.adapters.river._utils import is_river_estimator


class _RiverDeepcopyCloner(BaseCloner):
    """Clone River estimators via deepcopy.

    River models do not implement sklearn-style ``get_params``; this plugin
    allows ``RiverRegressor.clone()`` to produce independent River model copies.
    """

    def _check(self, obj):
        return is_river_estimator(obj)

    def _clone(self, obj):
        return deepcopy(obj)
