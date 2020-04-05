from skpro.metrics.proba_loss_cont import LogLossClipped
from skpro.metrics.proba_scorer import ProbabilisticScorer


class DefaultScoreMixin:
    "Mixin class for all end-used probabilistic estimators"
    
    def score(self, X, y):
        
        """Returns the log loss of the prediction.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix instead, shape = (n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for the estimator.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        Returns
        -------
        score : float
        """

        return ProbabilisticScorer(LogLossClipped())(self, X, y)


    