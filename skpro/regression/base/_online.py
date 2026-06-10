"""Base class for online (point-prediction) regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["patelchaitany"]

from skpro.regression.base._base import BaseProbaRegressor


class BaseOnlineRegressor(BaseProbaRegressor):
    """Base class for online regressors with point predictions only.

    Online regressors support incremental ``update`` after ``fit``, but do not
    implement probabilistic prediction methods such as ``predict_proba``.

    Probabilistic output should be provided by a meta-estimator wrapping
    online regressors, for example ``BaggingRegressor``.

    Input and output data types are handled by the parent class via skpro's
    datatype conversion layer; ``_fit``, ``_update``, and ``_predict`` receive
    inner representations (by default ``pd.DataFrame``).
    """

    _tags = {
        "object_type": "regressor_online",
        "estimator_type": "regressor",
        "capability:update": True,
        "capability:pred_int": False,
        "capability:multioutput": False,
        "capability:missing": False,
        "capability:survival": False,
    }

    def __init__(self):
        super().__init__()

    def _predict(self, X):
        """Predict labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as ``X``
            labels predicted for ``X``
        """
        raise NotImplementedError("abstract method")
