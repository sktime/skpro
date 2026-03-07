"""Base class for outlier detection using probabilistic regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd

from skpro.base import BaseEstimator


class BaseOutlierDetector(BaseEstimator):
    """Base class for outlier detection using probabilistic regressors.

    This class provides a pyod-compatible interface for outlier detection
    based on probabilistic regression models. It follows the pyod API with
    methods like fit, decision_function, and predict.

    Parameters
    ----------
    regressor : skpro probabilistic regressor
        A fitted or unfitted probabilistic regressor implementing the
        skpro BaseProbaRegressor interface.
    contamination : float, default=0.1
        The proportion of outliers in the dataset. Used to determine the
        threshold for binary classification in predict method.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data. The higher the score, the
        more abnormal the sample.
    threshold_ : float
        The threshold used to determine outliers. Samples with decision scores
        above this threshold are classified as outliers.
    """

    _tags = {
        "object_type": "outlier_detector",
        "estimator_type": "outlier_detector",
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, regressor, contamination=0.1):
        self.regressor = regressor
        self.contamination = contamination
        super().__init__()

    def fit(self, X, y=None):
        """Fit the outlier detector on training data.

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            Training feature data
        y : pandas DataFrame, pandas Series, or numpy array, default=None
            Training target data. If None, unsupervised outlier detection
            is performed based on the distribution of X.

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Convert inputs to pandas if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is not None and not isinstance(y, (pd.DataFrame, pd.Series)):
            y = pd.Series(y)

        # Fit the regressor if not already fitted
        if not self.regressor._is_fitted:
            if y is None:
                raise ValueError("Target variable y is required for fitting.")
            self.regressor.fit(X, y)

        # Calculate decision scores on training data
        self.decision_scores_ = self._compute_decision_scores(X, y)

        # Calculate threshold based on contamination
        self.threshold_ = np.percentile(
            self.decision_scores_, 100 * (1 - self.contamination)
        )

        self._is_fitted = True
        return self

    def decision_function(self, X, y=None):
        """Compute anomaly scores for samples.

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            Test feature data
        y : pandas DataFrame, pandas Series, or numpy array, default=None
            Test target data. Required for supervised outlier detection.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            Anomaly scores for each sample. Higher scores indicate
            more anomalous samples.
        """
        # Convert inputs to pandas if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is not None and not isinstance(y, (pd.DataFrame, pd.Series)):
            y = pd.Series(y)

        return self._compute_decision_scores(X, y)

    def predict(self, X, y=None):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            Test feature data
        y : pandas DataFrame, pandas Series, or numpy array, default=None
            Test target data. Required for supervised outlier detection.

        Returns
        -------
        is_outlier : numpy array of shape (n_samples,)
            Binary labels: 0 for inliers, 1 for outliers
        """
        scores = self.decision_function(X, y)
        return (scores > self.threshold_).astype(int)

    def _compute_decision_scores(self, X, y=None):
        """Compute decision scores for samples.

        This method should be implemented by subclasses.

        Parameters
        ----------
        X : pandas DataFrame
            Feature data
        y : pandas DataFrame or pandas Series, default=None
            Target data

        Returns
        -------
        scores : numpy array of shape (n_samples,)
            Decision scores for each sample
        """
        raise NotImplementedError("Subclasses must implement _compute_decision_scores")
