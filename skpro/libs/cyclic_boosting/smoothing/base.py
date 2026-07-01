"""
Base classes for smoothers
"""

import numpy as np
import sklearn.base as sklearnb

from skpro.libs.cyclic_boosting.utils import bin_steps


class AbstractBinSmoother(sklearnb.BaseEstimator, sklearnb.RegressorMixin):
    """**Abstract base class** for smoothers acting on bins.

    Please implement the methods ``fit`` and ``predict``.
    """

    inc_fitting = False
    supports_pandas = False


class SetNBinsMixin(object):
    """Mixin class for smoothers working on bins that saves binning
    information in ``fit`` to be available in ``predict``.
    """

    def set_n_bins(self, X_for_smoother):
        ndim = X_for_smoother.shape[1] - 2
        if ndim < 1:
            raise ValueError("Smoothers need at least three columns!")

        self.ndim_ = ndim
        self.bin_weights_ = X_for_smoother[:, -2]
        self.n_bins_ = np.max(X_for_smoother[:, : self.ndim_], axis=0)
        self.n_bins_ = np.round(self.n_bins_)
        self.n_bins_ = np.asarray(self.n_bins_, dtype=int) + 1
        if ndim > 1:
            self._bin_steps = bin_steps(self.n_bins_)


__all__ = ["AbstractBinSmoother", "SetNBinsMixin"]
