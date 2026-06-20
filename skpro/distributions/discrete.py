"""Discrete categorical distribution."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class Discrete(BaseDistribution):
    """Discrete distribution, modeling categorical probability distributions.

    Parameters
    ----------
    probabilities : pd.DataFrame or np.ndarray
        Array of probabilities for each class. If 2D, rows are instances, columns are classes.
        If DataFrame, column names are the class labels, unless classes is provided.
    classes : array-like, optional
        Class labels corresponding to the probabilities. If probabilities is a DataFrame
        and classes is None, classes are taken from the column names.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
        Names of the variables the distribution is for (e.g. ['target']).
    """

    _tags = {
        "authors": "skpro-developers",
        "capabilities:exact": [],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
    }

    def __init__(self, probabilities, classes=None, index=None, columns=None):
        self.probabilities = probabilities
        self.classes = classes

        super().__init__(index=index, columns=columns)

        if isinstance(probabilities, pd.DataFrame):
            if classes is None:
                self._classes = probabilities.columns.values
            else:
                self._classes = np.asarray(classes)
            self._probabilities = probabilities.values
        else:
            self._probabilities = np.asarray(probabilities)
            if classes is not None:
                self._classes = np.asarray(classes)
            else:
                if self._probabilities.ndim > 0:
                    self._classes = np.arange(self._probabilities.shape[-1])
                else:
                    self._classes = np.array([0])

    def mode(self):
        """Return the mode of the distribution (class with highest probability).

        Returns
        -------
        pd.DataFrame or scalar
            The most likely class for each instance.
        """
        if self._probabilities.ndim == 0:
            return self._classes[0]

        idx = np.argmax(self._probabilities, axis=-1)
        mode_vals = self._classes[idx]



        # Since it's typically (N_instances, K_classes)
        # mode_vals is of shape (N_instances,)
        if np.isscalar(mode_vals) or mode_vals.ndim == 0:
            if hasattr(self, "index") and len(self.index) == 1:
                return pd.DataFrame([mode_vals], index=self.index, columns=self.columns)
            return mode_vals

        return pd.DataFrame(mode_vals, index=self.index, columns=self.columns)
