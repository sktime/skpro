import numpy as np

from sklearn.ensemble.bagging import BaseBagging
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

from skpro.distributions.distribution_base import DistributionBase
from skpro.distributions.component.variate import VariateInfos
from skpro.estimators.base import ProbabilisticEstimator
from skpro.utils import utils


class BaggedDistribution(DistributionBase):
    
    def __init__(self, distributions):

        if not utils.dim(distributions) > 1 :
            raise ValueError('bagged distribution must have more than 1 distributions')
        
        self.distributions = distributions
        dtype = DistributionBase.distributionsType(self.distributions)
        variateComponent = VariateInfos(size = self.distributions[0].variateSize())
        
        super().__init__(
                name = 'ensemble',
                dtype = dtype,
                variateComponent = variateComponent
                )
    
    def pdf_imp(self, X):
  
         # Average the predicted PDFs
         arr = np.array([d.pdf(X)
                    for distribution in self.distributions
                    for d in distribution])

         return np.mean(arr, axis=0)
     
    def point(self):
  
         # Average the predicted PDFs
         arr = np.array([d.point()
                    for distribution in self.distributions
                    for d in distribution])

         return np.mean(arr, axis=0)


class ProbabilisticBaggingRegressor(BaseBagging, ProbabilisticEstimator):
    
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 #oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):

        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score= False,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        if not isinstance(base_estimator, ProbabilisticEstimator):
            raise ValueError("estimator arg is not of skpro probabilistic estimator type")


    def predict_proba(self, X):
        """ Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        averaged predicted distributions of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : skpro bagged distribution class = [n_samples]
            The predicted bagged distributions.
        """

        # Ensure estimator were being fitted
        check_is_fitted(self, "estimators_features_")
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])

        # Parallel loop
        from sklearn.ensemble.base import _partition_estimators
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        def _parallel_predict_regression(estimators, estimators_features, X):
            """ Private function used to compute predictions within a job. """
            return [
                estimator.predict_proba(X[:, features])
                for estimator, features in zip(estimators, estimators_features)
            ]

        # Obtain predictions
        all_y_hat = [
            _parallel_predict_regression(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X
            ) for i in range(n_jobs)
        ]

        all_y_hat  = np.array(all_y_hat) .flatten()
             
        return BaggedDistribution(all_y_hat)
    
    
    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(default= None)
            
            
    def _set_oob_score(self, X, y):
        raise NotImplementedError()
